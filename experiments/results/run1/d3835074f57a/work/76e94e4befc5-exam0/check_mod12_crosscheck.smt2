(set-option :produce-models true)
; Cross-check only: this modulus (12) is what the task's stated
; "绑定的方法" literally says to encode, even though the claim under
; test in this exam item is about modulus 24. Kept for audit/discrepancy
; documentation, not used as the primary verdict.
(declare-const n Int)
(assert (>= n 1))
(assert (<= n 1000))
(assert (not (= (mod (- (* n n n) n) 12) 0)))
(echo "PART_A_MOD12_RESULT:")
(check-sat)
(get-model)
