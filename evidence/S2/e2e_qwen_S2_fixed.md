# Transcribed from the lab terminal output pasted in chat (not the raw tee file). Lab session S2, main at 6257b42.
# Ollama: library=CUDA, A100-SXM4-80GB. Model qwen3:8b. Embedder all-MiniLM-L6-v2 on cuda.
4. SEARCH (hybrid, top 3)
  [table] What is the maximum amount for a home loan?
    bm25   loan_policy, p.1 text | loan_policy, p.2 text | loan_policy, p.1 table
    dense  loan_policy, p.1 table | loan_policy, p.1 text | loan_policy, p.2 text
    hybrid loan_policy, p.1 text | loan_policy, p.1 table | loan_policy, p.2 text
  (before the parser fix the home loan table row ranked low)
5. ANSWER with ollama (qwen3:8b)
  Q[text]: minimum credit score for unsecured personal loans -> 700 [loan_policy, p.1]. 4.7s. cites: True
  Q[table]: maximum amount for a home loan -> 50000000 INR [loan_policy, p.1]. 0.4s. cites: True
  Q[image]: fraud cases in the third quarter -> 980 [fraud_report, p.2]. 0.5s. cites: True   <-- WRONG
  Q[table]: payment channel with most fraud loss -> UPI, 310 lakh INR [fraud_report, p.1]. 0.5s. cites: True
Done in 179.7s
## Failure case
Chart data is [900, 1100, 1250, 980] for Q1..Q4, so Q3 = 1250. OCR returns "1250 1100 980 900 Q1 Q2 Q3 Q4":
numbers and labels come out in different orders, the label-to-value link is lost, and the model picks the third number (980).
Before the fix the same question answered "Not found".
