import ResilientProtocol.Types
import ResilientProtocol.Kernel
import ResilientProtocol.Safety
import ResilientProtocol.Progress
import ResilientProtocol.Trace
import ResilientProtocol.Conformance
import ResilientProtocol.Examples
import ResilientProtocol.ConformanceExamples
import ResilientProtocol.Regression
import ResilientProtocol.Mutations

/-!
Authoritative import root for the pure resilient DiLoCo coordination kernel.

Transport, timers, buffer/process effects, floating-point arithmetic, and
production execution remain native/runtime authorities.  This package models
only deterministic coordination decisions over typed evidence identities.
-/
