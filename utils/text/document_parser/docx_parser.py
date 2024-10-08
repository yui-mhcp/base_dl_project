# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings

from .parser import parse_document

@parse_document.dispatch
def parse_docx(filename, ** kwargs):
    """ Parses `.docx` files and return the list of paragraphs in the form {'text' : str} """
    try:
        from docx import Document
    except ImportError as e:
        warnings.warn('Please install the `docx` library : `pip install python-docx`')
        return []
    
    doc = Document(filename)
    
    return [{'text' : p.text} for p in doc.paragraphs]
