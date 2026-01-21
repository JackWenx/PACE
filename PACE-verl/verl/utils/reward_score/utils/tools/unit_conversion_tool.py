import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
import pint
import asyncio

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UnitConversionTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict=None, tool_schema: OpenAIFunctionToolSchema=None):
        tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "unit_conversion",
                "description": "Performs precise physical unit conversions for a wide range of units, including length, mass, time, temperature, and complex compound units (e.g., velocity, acceleration). Use this for any calculation involving different units.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The numerical value to be converted. This must be a valid integer or float. Example: 10.5"
                        },
                        "source_unit": {
                            "type": "string",
                            "description": "The unit of the input 'value'. Use standard unit symbols. Supports compound units with '/' for division and '^' or '**' for powers. Examples: 'kg', 'km', 'm/s', 'km/h^2'."
                        },
                        "target_unit": {
                            "type": "string",
                            "description": "The desired unit to convert the value to. Must be dimensionally compatible with the source unit. Examples: 'g', 'mile', 'ft/s', 'm/s^2'."
                        }
                    },
                    "required": ["value", "source_unit", "target_unit"]
                }
            }
        })
        
        # Use an empty dict for config if not provided
        if config is None:
            config = {}

        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None: instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"response": "", "reward": []}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        try:
            value = parameters.get("value")
            source_unit = parameters.get("source_unit")
            target_unit = parameters.get("target_unit")
        except KeyError as e:
            print(f'Key does not exist for tool calling: {e}')
            return f"Key Error: {str(e)}", 0.0, {}
        
        try:
            ureg = pint.UnitRegistry(system='mks')
            value = float(value)

            # 1. 使用 ureg.Quantity() 是创建带有单位的量的最明确、最安全的方式
            quantity = ureg.Quantity(value, source_unit)
            result = quantity.to(target_unit)
            response = f"Unit parsed value: {float(result.magnitude)} {result.units}"
            self._instance_dict[instance_id]["response"] = response
            return response, 0.0, {}
             # return json.dumps({"result": result.magnitude, "unit": str(result.units)})
        except pint.errors.DimensionalityError as e:
            print('Dimensionality mismatch error')
            return f"Dimensionality mismatch error: {str(e)}", 0.0, {}
        except pint.errors.UndefinedUnitError as e:
            print(f'Unknown unit: {str(e)} {parameters}')
            return f"Unknown unit: {str(e)}", 0.0, {}
        except Exception as e:
            print(f'Unkown error: {e} {parameters}')
            return f"Error: {str(e)}", 0.0, {}


    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id].get('reward', '[]')

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


if __name__ == '__main__':

    parameters = {
        "value": 0.01,
        "source_unit": "km/s^2",
        "target_unit": "m/s^2"
    }

    async def main():
        unit_tool = UnitConversionTool()
        
        # First, create the instance to initialize it in the tool's dictionary
        instance_id = '1'
        await unit_tool.create(instance_id)

        # Now, execute the tool for the created instance
        response, reward, metrics = await unit_tool.execute(instance_id, parameters)
        
        print(f"Response: {response}")
        print(f"Reward: {reward}")
        print(f"Metrics: {metrics}")
    
    asyncio.run(main())
