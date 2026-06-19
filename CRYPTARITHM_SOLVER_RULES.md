# Cryptarithm Solver Rules

## Puzzle Format

```
AB<CD = EF
GH-IJ = KL
MN*OP = QRST
Determine the result for: UV`WX = ?
```

- Each letter is a **symbol** (special character like `!`, `@`, `#`, `\`, `'`, etc.)
- Each symbol maps to a **single digit (0-9)**
- The same symbol always maps to the same digit
- Two-character pairs (like `AB`) form a **2-digit number**: `digit(A)*10 + digit(B)`
- The middle character (`<`, `-`, `*`, `` ` ``) is the **operator**
- Right side of `=` is the **result** (1-4 characters of symbols)

---

## Two Puzzle Types

### Deduce (646 puzzles)
- Query operator **appears in the examples**
- You can directly see how the query operator behaves
- Easier: identify the formula from examples, apply to query

### Guess (158 puzzles)
- Query operator **does NOT appear in examples**
- Must deduce the digit map from example operators, then figure out the query operator's formula
- Harder: requires both map deduction AND formula identification for unseen operator

---

## The Four Formula Families

Every puzzle belongs to one of 4 families. All operators in a puzzle come from the **same family**.

### 1. DIRECT_MAXMIN (461 puzzles — 57%)
Inputs: `max(a,b)` and `min(a,b)` where a,b are the two 2-digit numbers.

| Formula | Count | Typical Width |
|---------|-------|---------------|
| max+min | 226 | 2-3 digit |
| max*min | 202 | 3-4 digit |
| max-min | 132 | 1-2 digit |
| max\|\|min (concat) | 131 | 4 digit |
| min\|\|max | 64 | 4 digit |
| (max*min)±1 | 113 | 3-4 digit |
| (max+min)±1 | 105 | 2-3 digit |
| max%min (modulo) | 38 | 1-2 digit |
| (max-min)±1 | 36 | 1-2 digit |

**Sign-prefix display variants** (same math, but result always shown with operator char prefix):
| Formula | Display | Count | Note |
|---------|---------|-------|------|
| max-min | sign-prefix + (max-min) | 86 | Same as max-min, just displayed with prefix always |
| (min-max)+1 | sign-prefix + abs(min-max+1) | 53 | = -(max-min-1), displayed with prefix |
| (min-max)-1 | sign-prefix + (max-min+1) | 12 | = -(max-min+1), displayed with prefix |

### 2. REV_AB (236 puzzles — 29%)
Inputs: `rev(a)` and `rev(b)` — reverse the digits of each number, then operate.

| Formula | Count | Typical Width |
|---------|-------|---------------|
| rev(rev(a)*rev(b)) | 155 | 3-4 digit |
| rev(rev(a)-rev(b)) | 151 | 1-2 digit |
| rev(rev(a)+rev(b)) | 149 | 2-3 digit |
| rev(rev(a)+rev(b)+1) | 27 | 2-3 digit |
| rev(rev(a)\|\|rev(b)) | 26 | 4 digit |
| rev(rev(b)-rev(a)) | 19 | 1-2 digit |
| rev(rev(a)*rev(b)±1) | 31 | 3-4 digit |
| rev(rev(a)+rev(b)-1) | 15 | 2-3 digit |

**Sign-prefix display variant:**
| Formula | Display | Count |
|---------|---------|-------|
| abs(rev(rev(a)-rev(b))) | sign-prefix + abs value | 10 | Same math as rev(rev(a)-rev(b)), always shown with prefix |

### 3. DIRECT_SIMPLE (90 puzzles — 11%)
Inputs: `a` and `b` directly (no max/min, no rev).

| Formula | Count | Typical Width |
|---------|-------|---------------|
| a-b | 70 | 1-2 digit |
| a\|\|b (concat) | 42 | 4 digit |
| a*b | 34 | 3-4 digit |
| a+b | 22 | 2-3 digit |
| (a+b)±1 | 28 | 2-3 digit |
| (a*b)±1 | 26 | 3-4 digit |
| a%b or b%a | 8 | 1-2 digit |
| b\|\|a | 3 | 4 digit |
| b-a | 2 | 1-2 digit |

### 4. REV_MAXMIN (17 puzzles — 2%)
Inputs: `max(rev(a),rev(b))` and `min(rev(a),rev(b))` — reverse first, then max/min.

| Formula | Count |
|---------|-------|
| rev(max(ra,rb)*min(ra,rb)) | 14 |
| rev(max(ra,rb)+min(ra,rb)) | 11 |
| rev(max(ra,rb)-min(ra,rb)) | 10 |
| rev(max(ra,rb)%min(ra,rb)) | 5 |
| rev(max(ra,rb)\|\|min(ra,rb)) | 2 |

---

## Key Rules to Remember

### Rule 1: Result Width Narrows the Formula

| Result Width | Possible Formulas |
|-------------|-------------------|
| **1 digit** | max-min, a-b, modulo — small differences only |
| **2 digits** | add, subtract, max-min, rev subtraction — most common width |
| **3 digits** | max+min, rev(rev(a)+rev(b)), multiply when one input is small |
| **4 digits** | **concat** or **multiply** — ONLY these produce 4 digits from 2-digit inputs |

**Critical**: If result is 4 digits, test concat FIRST (much more common than multiply for 4-digit).

### Rule 2: Concat Detection (4-digit results)

For concat `a||b`: the result is literally the digits of `a` followed by digits of `b`.
```
25<78 → if < means a||b: result = 2578
      → if < means b||a: result = 7825  
      → if < means max||min: result = 7825
      → if < means min||max: result = 2578
```

**How to detect**: If the 4-digit result's first two digits match one input and last two match the other → it's concat.

### Rule 3: Sign-Prefix Display

Some results are displayed with the operator character as the first character of the result:
```
AB-CD = -EF    ← the '-' at start of result IS the operator character
```

This is a **display convention**, not a negative formula. It means:
- Compute the formula normally (e.g., `max-min`)
- The result body (after the prefix) is the absolute value: `abs(result)`
- The operator char is prepended as a display marker

**When does sign-prefix appear?**
- Some formulas ALWAYS display with sign-prefix (86 cases of `max-min` variant, 10 cases of `rev` variant)
- `a-b` shows sign-prefix about 52% of the time (when `a < b`, so `a-b` is negative)
- `rev(rev(a)-rev(b))` shows sign-prefix about 52% of the time

**For the solver**: When you see sign-prefix, the underlying math is the same. Strip the prefix, read the body digits, and match against formula candidates. The sign-prefix just tells you the numeric value was negative.

### Rule 4: All Operators in a Puzzle Share the Same Family

If you identify ONE operator's family, you know the family for ALL operators in that puzzle.

Example: If operator `<` uses `max+min` (DIRECT_MAXMIN family), then operator `-` in the same puzzle MUST also be from DIRECT_MAXMIN (like `max-min`, `max*min`, `max||min`, etc.)

### Rule 5: Solving Strategy (Chain of Reasoning)

**Step 1: Width Filter**
Look at result widths for each operator. This eliminates most formulas immediately.

**Step 2: Family Detection**
Pick the equation with the most distinctive pattern:
- 4-digit result where first 2 digits = one input → concat → DIRECT_MAXMIN or DIRECT_SIMPLE
- Sign-prefix result → subtraction variant
- Very large 4-digit result (>5000) → likely multiply

**Step 3: Formula Confirmation**
Once you suspect a formula, verify it produces the correct result for ALL equations with that operator. If formula `f` works for equation 1 but not equation 2 of the same operator → wrong formula.

**Step 4: Digit Map Extraction**
Once you know the formula:
- From `AB op CD = result_digits`, you know `a = digit(A)*10 + digit(B)` and `b = digit(C)*10 + digit(D)`
- The result value confirms the mapping
- Cross-reference across equations to pin down each symbol's digit

**Step 5: Apply to Query**
Use the digit map to convert query symbols → numbers, apply the formula, convert result back to symbols.

### Rule 6: Reverse Operation (`rev`)

`rev(x)` reverses the digits of `x`:
```
rev(56) = 65
rev(30) = 3    (leading zero dropped)
rev(100) = 1   
```

For REV_AB family: first reverse both inputs, then operate, then reverse the result.
```
rev(rev(a) + rev(b)):
  a=56, b=78 → rev(65 + 87) = rev(152) = 251
```

### Rule 7: Common Traps

1. **Zero-padding**: `03` as a 2-digit number = 3, but as symbols `AB` where `A=0, B=3`
2. **Same symbol, different position**: Symbol `!` always maps to the same digit everywhere
3. **Result can have different width than inputs**: `23 * 45 = 1035` (2-digit inputs → 4-digit result)
4. **Modulo results**: `max%min` can give 0, which appears as single-digit `0`
5. **The `±1` variants**: `(max*min)+1` vs `(max*min)-1` — check both when close

---

## Operator Distribution

- Most puzzles have **2-3 distinct operators** (2 ops: 37%, 3 ops: 61%)
- Each puzzle has **3-5 example equations** (3 eqs: 29%, 4 eqs: 37%, 5 eqs: 34%)
- Each operator appears in **1-3 equations** within a puzzle

---

## Solving Priority (Most to Least Common)

When testing formulas for an unknown operator, try in this order:

**For 4-digit results**: max||min → max*min → rev(rev(a)*rev(b)) → a||b → min||max → (max*min)±1
**For 3-digit results**: max+min → rev(rev(a)+rev(b)) → max*min → (max+min)±1 → a*b
**For 2-digit results**: max+min → max-min → rev(rev(a)-rev(b)) → a-b → (max+min)±1 → rev(rev(a)+rev(b))
**For 1-digit results**: max-min → rev(rev(a)-rev(b)) → max%min

---

## For Guess Puzzles Specifically

The query operator is NOT in examples. Strategy:
1. Use example equations to **build the digit map** (you know example formulas from pattern matching)
2. Convert query symbols to numbers using the map
3. Use **result width** of the answer to narrow formula candidates
4. Test top candidates in priority order
5. The answer's symbol pattern may give hints (sign-prefix → subtraction variant)
