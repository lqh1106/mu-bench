# Business value contracts

The T-R4 rules retained by T-R5/JT7 add field-scoped format comparisons to the established itemized semantic judge, independent review and conflict arbitration. Tool-free and ablation protocols are unchanged.

The field rules are published in each tool's argument descriptions. Input processing, expected-operation checks, state assertions and projected business-state comparison use the same module, `business_values.py`. Only named fields in 24 scenarios are affected. Raw submitted arguments and actual outcomes remain in the evidence.

- Human-readable labels ignore letter case and surrounding/repeated whitespace. Words and object identities are not deleted or guessed.
- Currency codes are normalized to uppercase; different currencies remain distinct.
- English weekday case and HH:MM hour zero-padding are equivalent. Date, AM/PM and timezone are never inferred.
- Explicit finite encodings, such as original payment method / original_payment_method, share a comparison value; different refund channels do not.
- Album titles remain separate from playback scope. Full Evening Jazz is not automatically changed to Evening Jazz.
- Public participant IDs, resource identifiers, phone/passport digits, numbers, owners, business statuses and unknown fields are not globally normalized.

A historical interaction can be checked for JT6 compatibility without replaying the agent. Its original case, tool schema and semantic judgments remain preserved. A marker must state whether the score was unchanged, deterministic items were recomputed, or a new semantic review was actually run. A JT6 compatibility label is not evidence that the agent received the T-R4 interface.

Explicit local `HH:MM AM/PM` values normalize to the equivalent 24-hour `HH:MM` on the declared event-time fields. Bare `06:00` and `18:00` remain different; the case must supply the daypart. Historical interactions retain the information they actually received.
