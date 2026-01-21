# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4
import json
import ray
import ray.actor
import ray.util.multiprocessing
import asyncio
import random # Import the random module

from .base_tool import BaseTool
from .sandbox_utils import _process_single_case
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")

# --- Helper function moved outside the class ---
import os
import logging

# logger can be configured elsewhere in your application
logger = logging.getLogger(__name__)

def load_service_ips(ip_file_path: str) -> list:
    if not os.path.exists(ip_file_path):
        logger.error(f"Ip path not find: {ip_file_path}")
        return []

    urls = []
    try:
        with open(ip_file_path, 'r') as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:
                    continue
                if clean_line.startswith(('http://', 'https://')):
                    urls.append(clean_line)
                else:
                    full_url = f"http://{clean_line}:8080/execute"
                    urls.append(full_url)
    except IOError as e:
        logger.error(f"cannot read '{ip_file_path}': {e}")
        return []

    return urls

class PoolMode(Enum):
    ThreadMode = 1
    ProcessMode = 2

@ray.remote(concurrency_groups={"acquire": 5, "release": 10})
class TokenBucketWorker:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0
        self._semaphore = asyncio.Semaphore(rate_limit) # Use asyncio.Semaphore for async acquire

    @ray.method(concurrency_group="acquire")
    async def acquire(self): # Made async
        await self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        return self.current_count

class ExecutionWorker:
    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        return True

    async def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T: # Made async
        if self.rate_limit_worker:
            await self.rate_limit_worker.acquire.remote()
            try:
                return await fn(*fn_args, **fn_kwargs) # Await the async function
            except Exception as e:
                logger.warning(f"Error when executing search: {e}")
            finally:
                self.rate_limit_worker.release.remote()
        else:
            return await fn(*fn_args, **fn_kwargs) # Await the async function

def init_execution_pool(num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode):
    if mode == PoolMode.ThreadMode:
        return ray.remote(ExecutionWorker).options(max_concurrency=num_workers).remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
    else:
        raise NotImplementedError("Process mode is not implemented yet")

class SandboxFusionTool(BaseTool):
    """A tool for executing code using a pool of sandbox fusion instances with load balancing."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "python_interpreter",
                    "description": "Executes a complete, runnable Python code.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "A string containing full Python code."}},
                        "required": ["code"],
                    },
                }
            })

        if config is None:
            config = {
                # **MODIFIED**: Removed single URL, using a path to a file with multiple IPs
                "sandbox_fusion_url_path": "sandbox_ips.txt",
                "num_workers": 10,
                "enable_global_rate_limit": False,
                "rate_limit": 10,
                "default_timeout": 60,
                "default_language": "python"
            }
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.default_timeout = config.get("default_timeout", 30)
        self.default_language = config.get("default_language", "python")
        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_execution_pool(num_workers=self.num_workers, enable_global_rate_limit=self.enable_global_rate_limit, rate_limit=self.rate_limit, mode=PoolMode.ThreadMode)
        
        self.sandbox_fusion_url_path = config.get("sandbox_fusion_url_path", "")
        self.sandbox_fusion_urls = load_service_ips(self.sandbox_fusion_url_path)
        
        if not self.sandbox_fusion_urls:
            raise ValueError("Sandbox fusion URL list is empty. Please check the path.")

        # Thread-safe components for round-robin
        self._url_index = 0
        self._lock = threading.Lock()
        
        log_msg = f"Init SandboxFusionTool with {len(self.sandbox_fusion_urls)} sandbox URLs."
        logger.info(log_msg)

    def _get_next_url(self) -> str:
        """
        Atomically gets the next URL from the list in a round-robin fashion.
        """
        with self._lock:
            # Select the URL and update the index for the next call
            url = self.sandbox_fusion_urls[self._url_index]
            self._url_index = (self._url_index + 1) % len(self.sandbox_fusion_urls)
            return url

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "ground_truth": ground_truth, "reward": []}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)

        target_url = self._get_next_url()
        logger.info(f"Executing code on sandbox: {target_url}")
        
        result = await self.execution_pool.execute.remote(
            SandboxFusionTool.execute_code, 
            instance_id, code, timeout, language, url=target_url
        )

        return result, result, result.strip()

    @staticmethod
    async def execute_code(instance_id, code, timeout=30, language="python", url=None):
        if not url:
            raise ValueError("Execution URL was not provided to execute_code.")
            
        # This function can be async if _process_single_case supports it.
        # Assuming _process_single_case is synchronous, we run it in an executor.
        loop = asyncio.get_running_loop()
        result_status, metadata = await loop.run_in_executor(
            None,  # Use default executor
            _process_single_case, 0, None, None, url, code, timeout, language
        )

        return json.dumps(metadata['api_response'], ensure_ascii=False)

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

if __name__ == "__main__":
    async def main():
        # --- SETUP FOR TEST ---
        # Create a dummy IP file for the test to run.
        # In a real scenario, this file would contain your sandbox server IPs.
        ip_file_name = "/code/fengruixiang/Daily/ip_pool/sandbox_ips.txt"
        # with open(ip_file_name, "w") as f:
        #     f.write("http://10.100.62.94:8080/execute\n")
        #     f.write("http://10.100.62.95:8080/execute\n") # Example second IP
        #     f.write("http://10.100.62.96:8080/execute\n") # Example third IP

        if not ray.is_initialized():
            ray.init()

        try:
            print("🚀 Starting SandboxFusionTool load balancing test...")
            
            config = {
                "sandbox_fusion_url_path": ip_file_name,
                "num_workers": 3,
                "enable_global_rate_limit": False,
                "rate_limit": 10,
                "default_timeout": 60,
                "default_language": "python"
            }

            sample_code = """
def main():
    import socket
    # Print the hostname to see which container/server ran the code
    print(f"Code executed on host: {socket.gethostname()}")
if __name__ == "__main__":
    main()
"""
            print("🔧 Initializing SandboxFusionTool...")
            tool = SandboxFusionTool(config=config, tool_schema=None)

            # --- Execute multiple times to test load balancing ---
            print("\nExecuting code 3 times to observe round-robin behavior...")
            tasks = []
            for i in range(3):
                instance_id = await tool.create()
                parameters = {"code": sample_code}
                # Create and collect the asyncio tasks
                task = tool.execute(instance_id, parameters)
                tasks.append(task)
            
            # Wait for all executions to complete
            results = await asyncio.gather(*tasks)

            print("\n--- ✅ Execution Results ---")
            for i, (response, _, _) in enumerate(results):
                # The actual hostname will depend on your sandbox environment
                print(f"Run {i+1} Response: {json.loads(response)['run_result']['stdout']}")
            print("--------------------------\n")

        except Exception as e:
            print(f"An error occurred during the test: {e}")
        finally:
            print(" shutting down Ray.")
            ray.shutdown()
            # os.remove(ip_file_name) # Clean up the dummy file

    asyncio.run(main())