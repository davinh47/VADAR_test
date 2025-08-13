DM_PROMPT = """
You are given a question and an indexed dictionary of candidate answers.

Input:
question: {question}
answers: {answers}

Select ONE index to return using these rules:

1) First, identify all answers that are valid, logical and consistent with the question.
   - Valid = a finite numeric value (not None/NaN/±inf) or a non-empty string.
      - If the question is yes/no, the answer must be "yes" or "no" in text, or a boolean result
      - If the question is multiple-choice among listed options, the answer must match one of those options (text or a valid index depends on what the question asks).
   - Logical = the value is sensible given the question context (e.g. when the question asks for a count/ordinal/index, negative numbers are illogical)
      - A numeric answer is illogical when it differs from an explicit reference in the question by several orders of magnitude (e.g., an indoor furniture height of 50 m vs. a reference of 0.4 m)

2) Prefer NON-GUESS over GUESS:
   - If an answer's corresponding program contains the marker "is_guess = True", treat it as a guess.
   - When at least one non-guess has a valid logical answer, ignore all guesses.

3) If multiple candidates remain after step 2, check their corresponding programs for severe logical flaws or mismatches between the program's logic and the question's intent (e.g., ignoring specific object constraints, quantities, spatial qualifiers) that directly affect the correctness of the result. Only remove candidates if such issues are clearly present.

4) Tie-break: if multiple valid and logical candidates remain equivalent after step 3, return the highest index among them.

5) Fallbacks:
   - If no candidate is logical, return the highest index among those with a valid answer.
   - If none have a valid answer, return 0.

Wrap your choice in <index>...</index>.
"""