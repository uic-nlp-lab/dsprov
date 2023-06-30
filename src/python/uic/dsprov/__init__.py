"""Discharge summary *provenience of data* annotations.  The term *antecedent*
for this module is a span of text found in the discharge summary copied and
pasted from the first originating note.

"""
from .domain import *
from .reader import DischargeSummaryAnnotationStash
from .writer import ProvenanceWriter
from .match import *
from .assess import *
from .extract import *
from .app import *
from .cli import *
