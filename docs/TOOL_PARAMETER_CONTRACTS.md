# Seat proximity priority parameter

The `update_attendee_seat_requirement` API describes `proximity_priority` publicly:

- `primary`: sitting near the group takes precedence.
- `secondary`: group proximity is a preference subordinate to mandatory seat requirements.
- `none`: no group-proximity preference.

The schema also lists finite equivalent spellings: `group_proximity_first` maps to `primary`; `accessibility_over_group_proximity` and `accessibility_first` map to `secondary`; `no_proximity_preference` maps to `none`. These are general API meanings, not the expected choice for a particular user.

Execution normalizes only this named field in the accessible-seat scenario. Scoring normalizes the same field when comparing historical call arguments and final business state. Other parameters, participant identities, resource IDs, authorization, seat requirements, timestamps and original tool logs remain unchanged. Unknown strings and conflicting priorities are not accepted as equivalent to `secondary`.

A historical re-evaluation may apply this documented finite equivalence to already executed effects. It must retain the original schema and raw arguments; it must not claim that a newly expanded schema was delivered during the historical session. Correcting a comparison does not fabricate an execution or convert a denied call into a successful one.
