C2D Repository Template
=======================
## Table of Contents
1. [Repository Name](#repo_name)
2. [Repository files structure](#repo_file_structure)
3. [Recommendations for Improving Code Quality!](#recommendations_improving_code_quality)
    1. [Basic Recommendations](#basic_recommendations)
    2. [Flake8](#flake8)
    3. [isort](#isort)
    4. [Additional tips](#additional_tips)
4. [Avoiding Unnecessary Complexity in Your Code](#unnecessary_complexity)
4. [Final Considerations](#final_considerations)
5. [Contact](#contact)


<br><br>

<a name="repo_name"></a>
## Repository Name

Choose a repository name that clearly reflects the project's purpose. Here are some examples:
- efficient-language-models
- causal-reasoning-with-data
- user-centered-LLMs-evaluation

If your project encompasses multiple repositories, include a concise descriptor for each one. For example:
- efficient-language-models-summarization-task
- efficient-language-models-dataset-constructor

This approach ensures that each repository name is both descriptive and easily identifiable, facilitating better organization and navigation.

> **_NOTE:_**  Always use lowercase letters (unless it is an abbreviation of a name or term, such as "LLM") and separate words with hyphens (-).

<br><br>

<a name="repo_file_structure"></a>
## Repository files structure 

A well-organized repository structure enhances readability and maintainability. Below is a recommended file structure:

    .
    ├── data/                 # Contains links to datasets. If the repository is or will be public, you can upload the datasets directly here
    ├── images/               # Contains images (avoid uploading too many images to private repositories)
    ├── scripts/              # Python or other language scripts (e.g., shell scripts for running experiments)
    ├── src/                  # Project source code
    ├── utils/                # Contains common code that is reusable and independent of the application's core logic
    ├── .gitignore            # File extensions or patterns to be ignored by Git
    ├── LICENSE               # Repository license
    ├── main.py               # Python script that contains an easy-to-run code example of the project
    ├── README.md             # Project overview. Should include a simple tutorial of how to use the main.py file
    └── requirements.txt      # Dependencies (preferably use pip instead of conda environments)


**Best Practices for File Extensions:**
- **Markdown** (`.md`): Use for documentation, as it is the default format on GitHub.
- **Text** (`.txt`): Use for listing requirements or plain text notes.
- **PDF** (`.pdf`): Preferable for reports or papers over Microsoft Word format (`.docx`) to ensure consistency and accessibility.

<br><br>


<a name="recommendations_improving_code_quality"></a>
## Recommendations for Improving Code Quality! 

Imagine opening a scientific paper that's left-aligned like the image below (left). It can be frustrating and hard to read, prompting you to put the paper down immediately. Compare this to the same text justified on both sides (right).

![Left-aligned vs justified text](/images/left-aligned-vs-justified.png)

Similarly, poorly formatted or written code can be just as, if not more, difficult to read. This can cause readers to lose focus quickly and damage your reputation as a developer. Consider the two code examples below and choose which one you find easier to read. You'll likely prefer the second one! 😉

```python
def process_order(order):
    if order['quantity'] > 100 and order['price'] > 500 and order['status'] == 'confirmed' and order['customer_type'] == 'premium' and order['delivery_date'] > '2024-01-01':
        print("Processing high-priority premium order")
    else:
        print("Processing regular order")
```

```python
def process_order(order):
    if (
        order['quantity'] > 100 
        and order['price'] > 500 
        and order['status'] == 'confirmed'
        and order['customer_type'] == 'premium'
        and order['delivery_date'] > '2024-01-01'
    ):
        print("Processing high-priority premium order")
    else:
        print("Processing regular order")
```

Improving your code quality may initially seem unnecessary or tedious, but over time you'll discover that these practices help you write more organized, efficient, and maintainable code. Remember, organization is key in all aspects of life—and your code is no exception!


<a name="basic_recommendations"></a>
### **Here are some recommendations to enhance your code quality:** 

**1. Consistent Formatting:**
 - Indentation: Use consistent indentation (e.g., 4 spaces) to improve readability.
 - Line Length: Keep lines within a reasonable length (e.g., 99 characters) to make the code easier to read and review.

**2. Meaningful Naming:**
- Choose descriptive variable and function names that clearly convey their purpose.
- Avoid abbreviations that may be unclear to others.
- An example to illustrate the importance of giving meaningful names:
    - Not Meaningful:
          `x = 50`
  - Meaningful:
          `max_user_allowed = 50` 


**3. Modular Code:**
- Break down complex functions into smaller, reusable components.
- This makes your code easier to test, debug, and maintain.

**4. Comprehensive Documentation:**
- Include docstrings for modules, classes, and functions to explain their functionality.
- Maintain a comprehensive `README.md` to provide an overview and usage instructions.

**5. Version Control:**
- Use `.gitignore` to exclude unnecessary files from your repository.
- Commit changes frequently with clear and descriptive messages to track your project's evolution effectively.

<br>

These recommendations are grounded in [PEP 8](https://peps.python.org/pep-0008/) and [PEP 257](https://peps.python.org/pep-0257/), the official Style Guide and Docstring Conventions for Python code (yes, there are official guidelines on how your code should be implemented!). While it's beneficial to read through these resources, you can also leverage tools to automatically identify and address areas for improvement in your code. We recommend the following libraries:
- **Flake8** ([Library Documentation](https://flake8.pycqa.org/en/latest/)): Ensures your code adheres to PEP 8 style conventions by checking for syntax errors, undefined names, and other stylistic issues.
- **isort** ([Library Documentation](https://pycqa.github.io/isort/)): Automatically sorts your import statements, organizing them into a consistent and readable order.

<br>

<a name="flake8"></a>
### Flake8 
**Flake8** is a powerful tool for enforcing style guidelines. It scans your code to identify deviations from PEP 8, such as improper indentation, excessive line lengths, and unused imports. By integrating Flake8 into your development workflow, you can maintain clean and consistent code, making it easier to read and maintain.

Key Features:
- **Syntax Checking:** Detects syntax errors that could cause your code to fail.
- **Style Enforcement:** Ensures adherence to PEP 8 guidelines, promoting uniform coding practices.
- **Plugin Support:** Extensible with plugins to add more checks or customize existing ones.

Usage Example:

```shell
flake8 your_script.py
```

Running this command will output any style violations or errors found in `your_script.py`, allowing you to address them promptly.

<br>

<a name="isort"></a>
### isort 
**isort** focuses specifically on the organization of import statements. It automatically sorts imports alphabetically and separates them into sections (standard library, third-party, and local imports), ensuring that your import statements are both orderly and compliant with best practices. This not only enhances readability but also helps prevent merge conflicts and import-related errors.

Key Features:
- **Automatic Sorting:** Organizes imports alphabetically and by category.
- **Customization:** Allows configuration to match specific project requirements.

Usage Example:

isort is very easy to use. You can sort the imports in a Python file by running the following command in your terminal:

```shell
isort your_script.py
```

After running the command, save the file to apply the sorted imports.

**Example of isort in Action:**

_Before isort:_
```python
import os
import sys
import requests
from mymodule import myfunction
import numpy as np
```

_After isort:_

```python
import os
import sys

import numpy as np
import requests

from mymodule import myfunction
```

In this example, isort has organized the imports into three distinct sections:
- **Standard Library Imports:** os, sys
- **Third-Party Imports:** numpy, requests
- **Local Application Imports:** mymodule

This separation improves readability and maintainability of your code by clearly distinguishing between different types of dependencies.

<br>

<a name="additional_tips"></a>
### Additional Tips:
- **Editor Integration:** Most modern code editors and IDEs, such as Visual Studio Code, support integrations for Flake8 and isort. These integrations provide real-time feedback as you write code, helping you adhere to style guidelines effortlessly. You can find easy setup tutorials with a simple search on the internet.
- **Formatters**: There are libraries, such as **black** ([library documentation](https://pypi.org/project/black/)),  that automatically format your code according to predefined rules. While these tools can simplify the formatting process, we encourage you to also develop an understanding of what constitutes high-quality code by manually writing and formatting your code.
- **Configuration Files:** Customize the behavior of Flake8 and isort by adding configuration files like `.flake8` or `pyproject.toml`. Tailoring these files allows you to adjust the tools to meet your project's specific needs. For example, Flake8 enforces a PEP 8 rule that raises an error if a line exceeds 79 characters. We recommend setting this limit to 99 characters. Refer to the NOTE below for detailed guidelines (retrieved from PEP 8) on maximum line length.

> **_NOTE:_** Limiting the required editor window width makes it possible to have several files open side by side, and works well when using code review tools that present the two versions in adjacent columns. ...  Some teams strongly prefer a longer line length. For code maintained exclusively or primarily by a team that can reach agreement on this issue, it is okay to increase the line length limit up to 99 characters, provided that comments and docstrings are still wrapped at 72 characters. ([PEP 8 Source](https://peps.python.org/pep-0008/#maximum-line-length))



<br><br>

<a name="unnecessary_complexity"></a>
## Avoiding Unnecessary Complexity in Your Code

Simplicity is the hallmark of effective and maintainable code. Introducing complex structures without a clear necessity can make your code harder to understand, debug, and extend. Strive to keep your implementations as straightforward as possible, using advanced constructs like classes only when they provide clear benefits.

**Why Avoid Unnecessary Complexity?**
1. **Readability:** Simple code is easier to read and comprehend, making it accessible to more developers, including your future self.
2. **Maintainability:** Less complex code reduces the likelihood of bugs and makes it easier to update or modify features.
3. **Performance:** Overly intricate structures can introduce performance overheads without tangible benefits.
4. **Collaboration:** Clear and simple code fosters better collaboration among team members, as it’s easier to onboard new contributors.

One of the most common pitfalls developers face is the unnecessary use of classes. Utilizing classes when they aren’t needed can complicate your codebase without providing meaningful advantages. To maintain simplicity and enhance code clarity, it’s essential to understand when to employ classes and when alternative approaches are more appropriate. Let’s explore the appropriate scenarios for using classes and identify situations where simpler constructs would be more effective.

**When to Use Classes?**

Classes are powerful tools in object-oriented programming, enabling encapsulation, inheritance, and polymorphism. However, they should be used carefully:

- Use Classes When:
  - You need to model real-world entities with attributes and behaviors.
  - You require multiple instances sharing common properties and methods.
  - You want to leverage inheritance to extend functionality.
  - Encapsulation is necessary to protect and manage data integrity.

- Avoid Classes When:
  - The functionality can be effectively handled with functions and simple data structures.
  - You’re writing small scripts or modules where object-oriented features offer no clear advantage.
  - Introducing a class would add unnecessary layers without improving code clarity or reusability.

**Example: Using Functions Over Classes**

**Overcomplicated Approach with Classes:**
```python
class OrderProcessor:
    def __init__(self, order):
        self.order = order

    def is_high_priority(self):
        return (
            self.order['quantity'] > 100 and
            self.order['price'] > 500 and
            self.order['status'] == 'confirmed' and
            self.order['customer_type'] == 'premium' and
            self.order['delivery_date'] > '2024-01-01'
        )

    def process(self):
        if self.is_high_priority():
            print("Processing high-priority premium order")
        else:
            print("Processing regular order")

def main():
    order = {
        'quantity': 150,
        'price': 600,
        'status': 'confirmed',
        'customer_type': 'premium',
        'delivery_date': '2024-02-01'
    }
    processor = OrderProcessor(order)
    processor.process()

if __name__ == "__main__":
    main()
```

**Simplified Approach with Functions:**

```python
def is_high_priority(order):
    return (
        order['quantity'] > 100 and
        order['price'] > 500 and
        order['status'] == 'confirmed' and
        order['customer_type'] == 'premium' and
        order['delivery_date'] > '2024-01-01'
    )

def process_order(order):
    if is_high_priority(order):
        print("Processing high-priority premium order")
    else:
        print("Processing regular order")

def main():
    order = {
        'quantity': 150,
        'price': 600,
        'status': 'confirmed',
        'customer_type': 'premium',
        'delivery_date': '2024-02-01'
    }
    process_order(order)

if __name__ == "__main__":
    main()

```

**Comparison:**
- **Clarity:** The function-based approach is more straightforward, making it easier to understand the flow.
- **Efficiency:** Fewer lines of code and no need to manage class instances.
- **Maintainability:** Simpler functions are easier to test and modify individually.


By avoiding unnecessary complexity and using advanced structures like classes only when they add clear value, you create code that is easier to read, maintain, and scale. Emphasize simplicity in your coding practices to enhance overall code quality and foster a more efficient development process.

<br><br>

<a name="final_considerations"></a>
## Final considerations 

This template is recommended for projects developed in C2D. Although it can be customized to fit your specific needs, following this structure will make the organization's repositories more comprehensible to external contributors and the wider community.

<br><br>

<a name="contact"></a>
## Contact
If you have any questions or suggestions about any of the topics covered by this template, please do not hesitate to contact us:

Leandro Mugnaini: leandromugnaini@alumni.usp.br

Daniel Lawand: daniel.lawand@alumni.usp.br
