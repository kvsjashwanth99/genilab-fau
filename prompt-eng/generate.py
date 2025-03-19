import requests
import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_config():
    """
    Load config file looking into multiple locations
    """
    config_locations = [
        "./_config",
        "prompt-eng/_config",
        "../_config"
    ]
    
    # Find CONFIG
    config_path = None
    for location in config_locations:
        if os.path.exists(location):
            config_path = location
            break
    
    if not config_path:
        raise FileNotFoundError("Configuration file not found in any of the expected locations.")
    
    # Load CONFIG
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()


def create_payload(model, prompt, target="ollama", **kwargs):
    """
    Create the Request Payload in the format required by the Model Server
    """
    payload = None
    if target == "ollama":
        payload = {
            "model": model,
            "prompt": prompt, 
            "stream": False,
        }
        if kwargs:
            payload["options"] = {key: value for key, value in kwargs.items()}

    elif target == "open-webui":
        payload = {
            "model": model,
            "messages": [ {"role" : "user", "content": prompt } ]
        }

    else:
        print(f'!!ERROR!! Unknown target: {target}')
    return payload


def model_req(payload=None):
    """
    Issue request to the Model Server
    """
    try:
        load_config()
    except:
        return -1, f"!!ERROR!! Problem loading prompt-eng/_config"

    url = os.getenv('URL_GENERATE', None)
    api_key = os.getenv('API_KEY', None)
    delta = response = None

    headers = dict()
    headers["Content-Type"] = "application/json"
    if api_key: headers["Authorization"] = f"Bearer {api_key}"

    # Send out request to Model Provider
    try:
        start_time = time.time()
        response = requests.post(url, data=json.dumps(payload) if payload else None, headers=headers)
        delta = time.time() - start_time
    except:
        return -1, f"!!ERROR!! Request failed! You need to adjust prompt-eng/config with URL({url})"

    # Checking the response and extracting the 'response' field
    if response is None:
        return -1, f"!!ERROR!! There was no response (?)"
    elif response.status_code == 200:
        result = ""
        delta = round(delta, 3)

        response_json = response.json()
        if 'response' in response_json: ## ollama
            result = response_json['response']
        elif 'choices' in response_json: ## open-webui
            result = response_json['choices'][0]['message']['content']
        else:
            result = response_json 
        
        return delta, result
    elif response.status_code == 401:
        return -1, f"!!ERROR!! Authentication issue. You need to adjust prompt-eng/config with API_KEY ({url})"
    else:
        return -1, f"!!ERROR!! HTTP Response={response.status_code}, {response.text}"
    return


def run_experiment(prompts, models, complexities, strategies, techniques, hyperparameters):
    """
    Run multi-variation parametric testing and collect results
    """
    results = []

    for prompt in prompts:
        for model in models:
            for complexity in complexities:
                for strategy in strategies:
                    for technique in techniques:
                        for hyperparameter in hyperparameters:
                            payload = create_payload(
                                model=model,
                                prompt=prompt,
                                target="ollama",
                                temperature=hyperparameter["temperature"],
                                num_ctx=hyperparameter["num_ctx"],
                                num_predict=hyperparameter["num_predict"]
                            )
                            time_taken, response = model_req(payload=payload)
                            if time_taken != -1:
                                results.append({
                                    "Prompt": prompt,
                                    "Prompt Complexity": complexity,
                                    "Prompt Strategy": strategy,
                                    "Prompt Eng Technique": technique,
                                    "Model": model,
                                    "Temperature": hyperparameter["temperature"],
                                    "Context Window": hyperparameter["num_ctx"],
                                    "Time": time_taken,
                                    "Executed Tokens": len(response.split()),  # Approximate token count
                                    "Quality of Result": "High"  # Placeholder, replace with actual evaluation
                                })

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel("experiment_results.xlsx", index=False)
    return df


def generate_graphs(df):
    """
    Generate graphs for the research report
    """
    # Set Seaborn style
    sns.set(style="whitegrid")

    # Graph 1: Impact of Prompt Complexity on Time Taken
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Prompt Complexity", y="Time", data=df)
    plt.title("Impact of Prompt Complexity on Time Taken")
    plt.xlabel("Prompt Complexity")
    plt.ylabel("Time Taken (seconds)")
    plt.savefig("complexity_vs_time.png")
    plt.close()

    # Graph 2: Impact of Temperature on Time Taken
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Temperature", y="Time", hue="Model", data=df)
    plt.title("Impact of Temperature on Time Taken")
    plt.xlabel("Temperature")
    plt.ylabel("Time Taken (seconds)")
    plt.savefig("temperature_vs_time.png")
    plt.close()

    # Graph 3: Impact of Model on Response Quality
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="Time", hue="Prompt Complexity", data=df)
    plt.title("Impact of Model on Response Quality")
    plt.xlabel("Model")
    plt.ylabel("Time Taken (seconds)")
    plt.savefig("model_vs_quality.png")
    plt.close()


###
### DEBUG
###

if __name__ == "__main__":
    # Define test parameters
    prompts = ["What is the capital of France?", "Explain quantum mechanics in simple terms."]
    models = ["llama3.2:latest", "mistral:7b"]  # Two models for comparison
    complexities = ["SIMPLE", "MID", "HIGH"]
    strategies = ["simple prompting", "level-1 automation", "level-2 automation"]
    techniques = ["CoT", "Self-Ask"]
    hyperparameters = [
        {"temperature": 0.7, "num_ctx": 2048, "num_predict": 100},
        {"temperature": 1.0, "num_ctx": 4096, "num_predict": 200}
    ]

    # Run experiment
    results_df = run_experiment(prompts, models, complexities, strategies, techniques, hyperparameters)

    # Generate graphs
    generate_graphs(results_df)
    print("Graphs generated successfully!")
