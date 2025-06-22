# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import re

from omegaconf import DictConfig

from file_parser import extract_code_blocks
from copy import deepcopy
from typing import Dict, List, Any

from modelscope_agent.agent import Runtime
from modelscope_agent.agent.code.base import Code
from modelscope_agent.llm.llm import LLM
from modelscope_agent.llm.utils import Message
from modelscope_agent.tools.filesystem_tool import FileSystemTool
from modelscope_agent.tools.split_task import SplitTask


class Coding(Code):
    """Split task and begin coding"""

    _frontend_prompt = """* **Overall Style:** Consider magazine-style, publication-style, or other modern web design styles you deem appropriate. The goal is to create a page that is both informative and visually appealing, like a well-designed digital magazine or in-depth feature article.

* **Hero Section (Optional but Strongly Recommended):** If you think it's appropriate, design an eye-catching Hero section. It can include a main headline, subtitle, an engaging introductory paragraph, and a high-quality background image or illustration.

* **Typography:**
  * Carefully select font combinations (serif and sans-serif) to enhance the reading experience.
  * Use different font sizes, weights, colors, and styles to create a clear visual hierarchy.
  * Consider using refined typographic details (such as drop caps, hanging punctuation) to enhance overall quality.
  * Font Awesome has many icons - choose appropriate ones to add visual interest and playfulness.

* **Color Scheme:**
  * Choose a color palette that is both harmonious and visually impactful.
  * Consider using high-contrast color combinations to highlight important elements.
  * Explore gradients, shadows, and other effects to add visual depth.

* **Layout:**
  * Use a grid-based layout system to organize page elements.
  * Make full use of negative space (whitespace) to create visual balance and breathing room.
  * Consider using cards, dividers, icons, and other visual elements to separate and organize content.

* **Tone:** Overall style should be refined, creating a sense of sophistication.

* **Data Visualization:**
  * Design one or more data visualization elements to showcase key concepts of Naval's thinking and their relationships.
  * Consider using mind maps, concept relationship diagrams, timelines, or thematic clustering displays.
  * Ensure visualization design is both beautiful and insightful, helping users intuitively understand the overall framework of Naval's thought system.
  * Use Mermaid.js to implement interactive charts that allow users to explore connections between different concepts.

**Technical Specifications:**

*   Use HTML5、Font Awesome、Tailwind CSS and necessary JavaScript。
    *   Font Awesome: [https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css](https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css)
    *   Tailwind CSS: [https://cdn.staticfile.org/tailwindcss/2.2.19/tailwind.min.css](https://cdn.staticfile.org/tailwindcss/2.2.19/tailwind.min.css)
    *   Font for non-Chinese: [https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap](https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap)
    *   `font-family: Tahoma,Arial,Roboto,"Droid Sans","Helvetica Neue","Droid Sans Fallback","Heiti SC","Hiragino Sans GB",Simsun,sans-self;`
    *   Mermaid: [https://cdn.jsdelivr.net/npm/mermaid@latest/dist/mermaid.min.js](https://cdn.jsdelivr.net/npm/mermaid@latest/dist/mermaid.min.js)

* Implement a complete dark/light mode toggle functionality that follows system settings by default and allows users to manually switch.
* Code structure should be clear and semantic, including appropriate comments.
* Implement complete responsiveness that must display perfectly on all devices (mobile, tablet, desktop).
"""  # noqa

    def __init__(self, config):
        super().__init__(config)
        self.llm = LLM.from_config(self.config)
        self.file_system = FileSystemTool(config)

    async def on_task_begin(self, runtime: Runtime, messages: List[Message]):
        await self.file_system.connect()

    async def generate_coding_tool_args(
            self, messages: List[Message]) -> Dict[str, Any]:
        """Manually generate coding tool arguments

        Args:
            messages: The messages of the architecture.

        Returns:
            The input arguments of the split-task tool.
        """
        arch_design = messages[2].content
        tasks, arch_design = extract_code_blocks(arch_design, target_filename='tasks.json')
        tasks = [t for t in tasks if t['filename'] == 'tasks.json'][0]
        tasks = tasks['code']
        if isinstance(tasks, str):
            tasks = json.loads(tasks)
        files = await self.file_system.list_files()

        sub_tasks = []
        for i, task in enumerate(tasks):
            system = task['system']
            query = task['query']
            coding_system  = (f'{system}\n\n'
                              f'The architectural design is {arch_design}\n\n'
                              f'The coding instruction of frontend: {self._frontend_prompt}\n\n'
                              f'The files existing on the filesystem is: {files}\n\n'
                              f'You must output your files with this format:\n\n'
                              f'```js:index.js\n'
                              f'... code ...\n'
                              f'```\n'
                              f'The `index.js` will be used to saving. '
                              f'You only need to generate/fix/update the files listed in the query, '
                              f'other modules will be handled in other tasks.\n'
                              f'You need consider the interfaces between your and other modules according to the architectural design.\n\n'
                              f'Now Begin:\n')
            coding_query = query
            task_arg = {
                'system': coding_system,
                'query': coding_query,
            }
            sub_tasks.append(task_arg)
        return {'tasks': sub_tasks}

    async def run(self, inputs, **kwargs):
        """Do a coding task.
        """
        config = deepcopy(self.config)
        sub_tasks = await self.generate_coding_tool_args(inputs)
        result = []
        max_retry = 3
        retry_cnt = 0
        failed_sub_tasks = []
        while retry_cnt < max_retry and len(sub_tasks['tasks']) > 0:
            split_task = SplitTask(config)
            tool_result = await split_task.call_tool(
                'split_task', tool_name='split_to_sub_task', tool_args=sub_tasks)
            single_task_results = [t.strip() for t in tool_result.split('SplitTask') if t.strip()]
            assert len(single_task_results) == len(sub_tasks['tasks'])
            new_sub_tasks = []
            failed_sub_tasks.clear()
            for i, (r, t) in enumerate(zip(single_task_results, sub_tasks['tasks'])):
                if 'failed' in r:
                    new_sub_tasks.append(sub_tasks['tasks'][i])
                    failed_sub_tasks.append(r)
                else:
                    result.append(r)
            retry_cnt += 1
            sub_tasks['tasks'] = new_sub_tasks

        result.extend(failed_sub_tasks)
        tool_result = 'SplitTask'+ 'SplitTask'.join(result)
        query = f'Code generation done, here is the result: {tool_result}'
        inputs.append(Message(role='user', content=query))
        return inputs
