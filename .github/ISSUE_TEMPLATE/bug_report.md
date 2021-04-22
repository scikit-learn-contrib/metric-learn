---
name: Reproducible bug report
about: Create a reproducible bug report. Not for support requests.
labels: 'bug'
---

#### Description
<!-- Describe your issue here.-->

#### Steps/Code to Reproduce
<!-- Please provide a **minimal** highlighted code example for
reproduction. (See https://help.github.com/articles/creating-and-highlighting-code-blocks/
for code blocks highlighting, and https://stackoverflow.com/help/mcve
for what is a minimal reproducible code.)

Example:
```python
from metric_learn import NCA
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

X, y = make_classification(random_state=0)
nca = NCA()
nca.fit(X, y)
```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. You can use ```ptb for python traceback formatting-->

#### Versions
<!-- Please run the following snippet and paste the output below.

import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)
import metric_learn; print("Metric-Learn", metric_learn.__version__)

(If the last statement returns "AttributeError: 'module' object has no attribute '__version__'", you can instead run this in a terminal:
$ pip show metric_learn | grep Version
)
-->
<!-- Thanks for contributing! -->

---
<!-- Issue Author: Don't delete this message to encourage other users to support your issue! -->
**Message from the maintainers**:

Impacted by this bug? Give it a 👍. We prioritise the issues with the most 👍.