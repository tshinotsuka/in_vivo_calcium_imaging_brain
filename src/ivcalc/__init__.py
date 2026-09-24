"""ivcalc — calcium imaging analysis for the AQP4 water intoxication work.

The analysis lives here and the scripts under scripts/ are thin command-line
wrappers around it. Anything computed in more than one place belongs in this
package: seven copies of the rolling baseline had become six implementations
before it existed, so a correction applied to the numbers did not reach the
figures drawn from them.
"""

__version__ = "0.2.0"

from . import events, io, traces, viz

__all__ = ["events", "io", "traces", "viz"]
