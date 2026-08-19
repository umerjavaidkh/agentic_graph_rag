"""Everything that turns ingested documents into answers.

Parsing a PDF into an intermediate representation, building the structural
(Axis-1) and semantic (Axis-2) graphs from it, and retrieving over the
result. The counterpart of `structured/`, which answers from the business
graph loaded out of tabular sources. Neither imports the other; both sit on
`shared/`.
"""
