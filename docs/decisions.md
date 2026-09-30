# Decision log

## D1: Study area uses two boundary vintages (2026-09-30)

**Context.** The downloaded boundaries (Tank Information System, repo last updated April 2016)
show the undivided Nalgonda district: 14,234 km², 59 mandals. Telangana reorganised its
districts on 11 October 2016, splitting Nalgonda into Nalgonda, Suryapet and Yadadri
Bhuvanagiri. The current Nalgonda boundary (geoBoundaries, representing 2021) measures
7,180 km² (official figure: 7,122 km²).

**Decision.** Run all satellite processing once over the pre-2016 extent, which contains
99.2% of the 2021 district. Report results for both boundaries, and show both on the dashboard.

**Consequences.**
- Comparisons with current official crop statistics use the 2021 boundary.
- Pre-2016 results allow comparison with older statistics published for the undivided district.
- The 0.8% mismatch between the two sources (boundary slivers) is documented, not hidden.