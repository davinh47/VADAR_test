DM_PROMPT = """
You are given a question and an indexed dictionary of candidate answers.

Input:
question: {question}
answers: {answers}

Select ONE index to return using these rules:

0) First, identify all answers that are logical and consistent with the question.
   - Logical = not None/NaN/±inf and not an empty string; type matches the question (e.g., yes/no → "yes"/"no").

1) Prefer NON-GUESS over GUESS:
   - If an answer/program contains the marker "is_guess = True", treat it as a guess.
   - When at least one non-guess has a valid logical answer, ignore all guesses.

2) Among the remaining logical answers, compare their corresponding programs. Choose the one that:
   - does not contain major reasoning errors;
   - most closely aligns with the intent of the question (e.g., stable bbox selection, unit consistency, sensible use of depth, avoiding duplicate counting).

3) Tie-break: if multiple logical candidates remain equivalent, choose the one with the **HIGHEST index**.

4) Fallbacks:
   - If no candidate is logical, return 0.
   - If your chosen index would be 0 but answers[0] is not logical while some other index is, instead return the **HIGHEST index** among those with a logical answer.
   - If none have a logical answer, return 0.

Wrap your choice in <index>...</index>.
"""