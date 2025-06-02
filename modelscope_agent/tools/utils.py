from dataclasses import dataclass


@dataclass
class ToolField:

    name: str = None
    description: str = None
    input_schema: str = None