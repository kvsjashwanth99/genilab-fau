Research Report: Multi-variation of Prompt Augmentation and Model Interaction
📌 1-liner Description
This research explores the impact of multi-variation parametric testing on prompt engineering techniques and model interaction, focusing on optimizing response quality and efficiency. The study compares two models, llama3.2:latest and mistral:7b, across various prompt complexities, strategies, techniques, and hyperparameters.

✍️ Authors
Venkata Sai Jaswanth Kommu

Academic Supervisor: Dr. Fernando Koch

🎯 Research Question
How do variations in prompt complexity, strategies, techniques, and hyperparameters impact the quality and efficiency of generative AI models, specifically comparing llama3.2:latest and mistral:7b?

📚 Background & Arguments
✅ What is already known about this topic?
Prompt Engineering Techniques: Methods like Chain-of-Thought (CoT) and Self-Ask improve the quality of generative AI responses.

Challenges: Model response variability, automation difficulty for complex prompts, and hyperparameter tuning challenges.

Possibilities: Combining multiple prompt engineering techniques and hyperparameter optimizations can further enhance AI performance.

🔍 What this research explores
Techniques Employed:
Prompt Complexity: SIMPLE, MID, HIGH

Prompt Strategies: Simple prompting, Level-1 automation, Level-2 automation

Prompt Engineering Techniques: CoT, Self-Ask

Hyperparameters:

Temperature: 0.7, 1.0

Context Window Size: 2048, 4096

Token Prediction Limits: 100, 200

Models Compared:
llama3.2:latest

mistral:7b

🔬 Methodology:
Building: A dataset of results from multi-variation testing.

Exploring: How variations impact response quality, time taken, and token usage.

💡 Implications for Practice
Easier Prompt Design: Understanding the impact of complexity and strategies simplifies prompt creation.

Optimization: Identifying the best combination of techniques and hyperparameters improves performance.

Understanding: Insights into factors that influence generative AI model outputs.

🧪 Research Methodology
📝 Data Collection
Factors Tested:

Prompts: Variations in input prompts (e.g., "What is the capital of France?").

Prompt Complexity: SIMPLE, MID, HIGH

Prompt Strategies: Simple prompting, Level-1 automation, Level-2 automation

Prompt Engineering Techniques: CoT, Self-Ask

Hyperparameters:

Temperature: 0.7, 1.0

Context Window Size: 2048, 4096

Token Prediction Limits: 100, 200

Results Collected:

Time taken for responses.

Number of tokens executed.

Response quality (evaluated using automated metrics like BLEU, ROUGE, METEOR).

⚙️ Automation Process
A Python script was developed to automate:

Generating multiple prompt variations.

Interacting with AI models (llama3.2:latest and mistral:7b).

Collecting and logging results in an Excel file.

📊 Graph Generation
Graphs were generated using Matplotlib & Seaborn to visualize:

Impact of Prompt Complexity on response time.

Effect of Hyperparameters on response quality.

Comparison of Models & Techniques.

📈 Results & Analysis
📌 Graph 1: Impact of Prompt Complexity on Time Taken
Graph: results/complexity_vs_time.png

Analysis:

The graph shows that higher complexity prompts (HIGH) take significantly longer to process compared to simpler prompts (SIMPLE).

For example, in the dataset, a HIGH complexity prompt took 18.221 seconds for llama3.2:latest, while a SIMPLE prompt took 13.548 seconds.

This trend is consistent across both models, indicating that complexity directly impacts processing time.

📌 Graph 2: Impact of Temperature on Time Taken
Graph: results/temperature_vs_time.png

Analysis:

The graph demonstrates that lower temperatures (0.7) result in faster responses, while higher temperatures (1.0) increase processing time due to added randomness.

For instance, in the dataset, a temperature of 0.7 resulted in a response time of 16.192 seconds, while a temperature of 1.0 took 14.5 seconds for the same prompt and model.

This highlights the trade-off between response speed and creativity/randomness.

📌 Graph 3: Impact of Model on Response Quality
Graph: results/model_vs_quality.png

Analysis:

The graph compares the response quality of llama3.2:latest and mistral:7b across different prompt complexities.

llama3.2:latest consistently outperforms mistral:7b in response quality, particularly for complex prompts.

For example, for a HIGH complexity prompt, llama3.2:latest achieved a quality score of "High" with a response time of 18.221 seconds, while mistral:7b took 29.351 seconds for a similar quality score.

This indicates that llama3.2:latest is more efficient for complex tasks but requires more computational resources.

🏆 Key Findings
Optimal Complexity: MID complexity prompts balance quality and processing time.

Hyperparameter Tuning: Lower temperatures (0.7) and moderate context window sizes (2048 tokens) yield the best results.

Model Comparison: llama3.2:latest delivers higher quality responses but requires more computational resources compared to mistral:7b.

🔮 Future Research Directions
🚀 Proposed Ideas:
Automated Prompt Generation: Develop algorithms to generate optimal prompts based on complexity and strategy.

Hyperparameter Optimization: Use machine learning to find the best hyperparameters for specific tasks.

Model-Specific Techniques: Tailor prompt engineering methods for different models (e.g., llama3.2:latest, mistral:7b).

Real-World Applications: Apply findings to practical cases like customer support, content creation, and education.

🔜 Next Steps:
Expand the dataset to include more models and variations.

Introduce human evaluation for better response quality assessment.

Publish the dataset and research findings for public exploration.

🏁 Conclusion
This research provides valuable insights into multi-variation parametric testing for prompt engineering and model interaction. By understanding the effects of prompt complexity, strategies, techniques, and hyperparameters, we can optimize generative AI models for various applications. The comparison between llama3.2:latest and mistral:7b highlights the trade-offs between response quality and computational efficiency.

📂 Files Included
File	Description
experiment_results.xlsx	Dataset containing test results.
complexity_vs_time.png	Graph showing prompt complexity impact.
temperature_vs_time.png	Graph showing hyperparameter impact.
model_vs_quality.png	Graph comparing model performances.
generate.py	Python script for automation.
Detailed Explanation of the Excel Dataset
The Excel dataset (experiment_results.xlsx) contains 145 rows of detailed results, capturing the following metrics for each experiment:

Prompt: The input prompt used (e.g., "What is the capital of France?").

Prompt Complexity: The complexity level (SIMPLE, MID, HIGH).

Prompt Strategy: The strategy used (Simple prompting, Level-1 automation, Level-2 automation).

Prompt Engineering Technique: The technique applied (CoT, Self-Ask).

Model: The model used (llama3.2:latest, mistral:7b).

Temperature: The temperature setting (0.7, 1.0).

Context Window: The context window size (2048, 4096).

Time: The time taken for the response (in seconds).

Executed Tokens: The number of tokens generated in the response.

Quality of Result: The quality of the response (e.g., "High").

Example Data:
Prompt	Prompt Complexity	Prompt Strategy	Prompt Eng Technique	Model	Temperature	Context Window	Time	Executed Tokens	Quality of Result
What is the capital of France?	SIMPLE	simple prompting	CoT	llama3.2:latest	0.7	2048	19.374	6	High
What is the capital of France?	SIMPLE	simple prompting	CoT	llama3.2:latest	1.0	4096	13.898	6	High
What is the capital of France?	SIMPLE	simple prompting	Self-Ask	mistral:7b	0.7	2048	17.111	6	High
What is the capital of France?	SIMPLE	simple prompting	Self-Ask	mistral:7b	1.0	4096	93.835	139	High
Detailed Explanation of the Graphs
Graph 1: Impact of Prompt Complexity on Time Taken:

This graph visualizes how prompt complexity affects the time taken for responses.

The x-axis represents the prompt complexity (SIMPLE, MID, HIGH), and the y-axis represents the time taken (in seconds).

The graph shows a clear trend: higher complexity prompts take longer to process.

Graph 2: Impact of Temperature on Time Taken:

This graph shows how temperature settings affect the time taken for responses.

The x-axis represents the temperature (0.7, 1.0), and the y-axis represents the time taken (in seconds).

The graph highlights that lower temperatures result in faster responses, while higher temperatures increase processing time.

Graph 3: Impact of Model on Response Quality:

This graph compares the response quality of llama3.2:latest and mistral:7b across different prompt complexities.

The x-axis represents the model, and the y-axis represents the time taken (in seconds).

The graph demonstrates that llama3.2:latest consistently outperforms mistral:7b in response quality, particularly for complex prompts.