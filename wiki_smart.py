import ollama
import wikipedia
import sys
import logging
import os
import json
import time
import concurrent.futures
from logging.handlers import RotatingFileHandler

def setup_logging(log_to_console=True, log_file="advanced_wiki_ollama.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def query_llm(model_name, prompt):
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error querying LLM: {str(e)}")
        return None

def extract_topics(model_name, user_query):
    prompt = f"""
I need you to analyze this query and extract all key topics that need research:
{user_query}

Format your response as a JSON list of strings containing only the key topics.
Example: ["topic1", "topic2", "topic3"]
"""
    response = query_llm(model_name, prompt)
    try:
        topics = json.loads(response)
        return topics
    except:
        logging.warning(f"Failed to parse topics JSON: {response}")
        # Fallback: try to extract topics with a simpler approach
        prompt = f"List the 3-5 most important topics in this query as a comma-separated list: {user_query}"
        response = query_llm(model_name, prompt)
        return [topic.strip() for topic in response.split(',')]

def evaluate_topic_granularity(model_name, topics, user_query):
    prompt = f"""
Evaluate these topics extracted from the query:
"{user_query}"

Topics: {json.dumps(topics)}

For each topic, classify it as:
1. TOO_BROAD: Needs to be broken down into subtopics
2. TOO_NARROW: Needs to be generalized
3. APPROPRIATE: Good granularity for research

Format your response as a JSON object with topics as keys and classifications as values.
Example: {{"topic1": "TOO_BROAD", "topic2": "APPROPRIATE"}}
"""
    response = query_llm(model_name, prompt)
    try:
        evaluation = json.loads(response)
        return evaluation
    except:
        logging.warning(f"Failed to parse evaluation JSON: {response}")
        return {topic: "APPROPRIATE" for topic in topics}  # Fallback

def refine_topics(model_name, topics, evaluation, user_query):
    refined_topics = []
    
    for topic, status in evaluation.items():
        if status == "TOO_BROAD":
            prompt = f"""
The topic '{topic}' is too general for effective research on the query:
"{user_query}"

Please break this down into 2-4 more specific subtopics that would be more useful for researching this query.
Format your response as a JSON list of strings.
"""
            response = query_llm(model_name, prompt)
            try:
                subtopics = json.loads(response)
                refined_topics.extend(subtopics)
            except:
                logging.warning(f"Failed to parse subtopics JSON: {response}")
                refined_topics.append(topic)  # Keep original as fallback
                
        elif status == "TOO_NARROW":
            prompt = f"""
The topic '{topic}' is too specific for finding general information.
Please suggest 1-2 broader topics that would encompass this specific topic while still being relevant to:
"{user_query}"

Format your response as a JSON list of strings.
"""
            response = query_llm(model_name, prompt)
            try:
                broader_topics = json.loads(response)
                refined_topics.extend(broader_topics)
            except:
                logging.warning(f"Failed to parse broader topics JSON: {response}")
                refined_topics.append(topic)  # Keep original as fallback
                
        else:  # APPROPRIATE
            refined_topics.append(topic)
            
    return refined_topics

def verify_topic_relevance(model_name, topics, user_query):
    prompt = f"""
Given the original query:
"{user_query}"

Evaluate how relevant each of these topics is:
{json.dumps(topics)}

Rate each topic's relevance on a scale of 1-10.
Format your response as a JSON object with topics as keys and scores as values.
Example: {{"topic1": 8, "topic2": 5}}
"""
    response = query_llm(model_name, prompt)
    try:
        relevance_scores = json.loads(response)
        return relevance_scores
    except:
        logging.warning(f"Failed to parse relevance scores JSON: {response}")
        return {topic: 8 for topic in topics}  # Fallback

def search_wikipedia(topic, num_sentences=8):
    try:
        logging.info(f"Searching Wikipedia for: '{topic}'")
        
        search_results = wikipedia.search(topic)
        
        if not search_results:
            logging.warning(f"No Wikipedia results found for '{topic}'")
            return f"No Wikipedia results found for '{topic}'."
        
        page_title = search_results[0]
        logging.info(f"Selected Wikipedia page: '{page_title}'")
        
        page_summary = wikipedia.summary(page_title, sentences=num_sentences)
        
        return f"Information about '{page_title}':\n\n{page_summary}"
    
    except wikipedia.exceptions.DisambiguationError as e:
        logging.warning(f"Disambiguation page found for '{topic}'. Selecting first option: '{e.options[0]}'")
        return search_wikipedia(e.options[0], num_sentences)
    
    except wikipedia.exceptions.PageError:
        logging.error(f"No specific Wikipedia page found for '{topic}'")
        return f"No specific Wikipedia page found for '{topic}'."
    
    except Exception as e:
        logging.error(f"An error occurred while searching Wikipedia: {str(e)}")
        return f"An error occurred while searching Wikipedia: {str(e)}"

def gather_information(topics):
    topic_information = {}
    
    # Use ThreadPoolExecutor for parallel Wikipedia searches
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_topic = {executor.submit(search_wikipedia, topic): topic for topic in topics}
        
        for future in concurrent.futures.as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                topic_information[topic] = future.result()
            except Exception as e:
                logging.error(f"Error gathering information for '{topic}': {str(e)}")
                topic_information[topic] = f"Failed to gather information: {str(e)}"
    
    return topic_information

def synthesize_information(model_name, topic_information, user_query):
    # Prepare the information for synthesis
    info_text = "\n\n".join([f"TOPIC: {topic}\n{info}" for topic, info in topic_information.items()])
    
    prompt = f"""
I've gathered information about several topics related to the query:
"{user_query}"

Please synthesize this information into a coherent knowledge base that addresses all aspects of the query.

{info_text}

Your synthesis should organize the information logically, identify connections between topics, and highlight the most relevant points for answering the query.
"""
    
    return query_llm(model_name, prompt)

def generate_response(model_name, synthesized_info, user_query):
    prompt = f"""
Based on this synthesized information:

{synthesized_info}

Please provide a comprehensive answer to the original query:
"{user_query}"

Your answer should be well-structured, balanced, and address all aspects of the query. Include specific examples where appropriate.
"""
    
    return query_llm(model_name, prompt)

def verify_response(model_name, generated_answer, user_query):
    prompt = f"""
Please verify this answer to the query:
"{user_query}"

ANSWER:
{generated_answer}

Check for:
1. Factual accuracy
2. Completeness (addresses all aspects of the query)
3. Balance (presents multiple perspectives fairly)
4. Logical coherence

If you find any issues, please provide corrections. If the answer is satisfactory, respond with "VERIFICATION: The answer is accurate and complete."
"""
    
    verification = query_llm(model_name, prompt)
    
    if "VERIFICATION: The answer is accurate and complete" in verification:
        return generated_answer
    else:
        # Use the verification feedback to improve the answer
        prompt = f"""
Based on this verification feedback:

{verification}

Please improve the following answer to the query:
"{user_query}"

ORIGINAL ANSWER:
{generated_answer}

Provide a revised answer that addresses the issues identified in the verification.
"""
        
        return query_llm(model_name, prompt)

def process_query(model_name, user_query):
    logging.info(f"Processing query: '{user_query}'")
    
    # Stage 1: Extract topics
    logging.info("Stage 1: Extracting topics")
    topics = extract_topics(model_name, user_query)
    logging.info(f"Extracted topics: {topics}")
    
    # Stage 2: Evaluate and refine topics
    logging.info("Stage 2: Evaluating topic granularity")
    evaluation = evaluate_topic_granularity(model_name, topics, user_query)
    logging.info(f"Topic evaluation: {evaluation}")
    
    logging.info("Refining topics based on evaluation")
    refined_topics = refine_topics(model_name, topics, evaluation, user_query)
    logging.info(f"Refined topics: {refined_topics}")
    
    # Verify topic relevance and filter out low-scoring topics
    relevance_scores = verify_topic_relevance(model_name, refined_topics, user_query)
    logging.info(f"Topic relevance scores: {relevance_scores}")
    
    final_topics = [topic for topic in refined_topics if relevance_scores.get(topic, 0) >= 6]
    logging.info(f"Final topics after relevance filtering: {final_topics}")
    
    # Stage 3: Gather information
    logging.info("Stage 3: Gathering information from Wikipedia")
    topic_information = gather_information(final_topics)
    
    # Stage 4: Synthesize information
    logging.info("Stage 4: Synthesizing information")
    synthesized_info = synthesize_information(model_name, topic_information, user_query)
    
    # Stage 5: Generate response
    logging.info("Stage 5: Generating response")
    response = generate_response(model_name, synthesized_info, user_query)
    
    # Stage 6: Verify response
    logging.info("Stage 6: Verifying response")
    verified_response = verify_response(model_name, response, user_query)
    
    logging.info("Query processing complete")
    return verified_response

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Advanced Wikipedia-Ollama Query System')
    parser.add_argument('--model', default='gemma3:1b', help='Ollama model to use')
    parser.add_argument('--log-file', default='advanced_wiki_ollama.log', help='Log file path')
    parser.add_argument('--console-log', action='store_true', help='Also log to console')
    args = parser.parse_args()
    
    setup_logging(log_to_console=args.console_log, log_file=args.log_file)
    
    logging.info(f"Starting advanced query system with Ollama model: {args.model}")
    print(f"Using Ollama model: {args.model}")
    print(f"Logging to file: {os.path.abspath(args.log_file)}")
    print("Enter your query (or 'exit' to quit):")
    
    while True:
        user_input = input("> ")
        
        if user_input.lower() in ['exit', 'quit']:
            logging.info("User requested to exit the application")
            break
        
        print("\nProcessing your query. This may take a few moments...\n")
        
        start_time = time.time()
        response = process_query(args.model, user_input)
        end_time = time.time()
        
        print("\nAnswer:")
        print(response)
        print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()