from dataclasses import dataclass


@dataclass
class RunStatus:

    should_stop: bool = False