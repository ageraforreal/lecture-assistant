import os
import sys

# корень проекта и папка tests доступны для импорта
HERE = os.path.dirname(__file__)
sys.path[:0] = [os.path.dirname(HERE), HERE]
