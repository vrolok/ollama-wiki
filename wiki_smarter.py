import ollama
import wikipedia
import sys
import logging
import os
import json
import time
import concurrent.futures
from logging.handlers import RotatingFileHandler
import re
import argparse

def setup_logging(log_to_console=True, log_file="advanced_wiki_ollama.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate logs
    logger.handlers = []
    
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

def extract_json(text):
    if not text:
        return None
        
    # Try to extract JSON from code blocks first
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Look for JSON-like structures with curly braces or square brackets
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = text
    
    # Try parsing the JSON directly
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Clean up common JSON formatting issues
        cleaned = re.sub(r'[\n\r\t]', '', json_str)
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON: {json_str[:100]}...")
            return None

def query_llm(model_name, prompt, max_retries=3, retry_delay=2):
    for attempt in range(max_retries):
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
            logging.error(f"Error querying LLM (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None

def extract_topics(model_name, user_query):
    prompt = f"""
Analyze this query and extract all key topics that need research:
{user_query}

Format your response as a JSON list of strings containing only the key topics.

Example: ["topic1", "topic2", "topic3"]

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
    response = query_llm(model_name, prompt)
    topics = extract_json(response)
    
    if not topics or not isinstance(topics, list):
        logging.warning("Failed to extract topics, using fallback topics")
        return ['Error Handling']
    
    return topics

def evaluate_topic_granularity(model_name, topics, user_query):
    prompt = f"""
Evaluate these topics extracted from the query:
{user_query}

Topics: {json.dumps(topics)}

For each topic, classify it as:
1. TOO_BROAD: Needs to be broken down into subtopics
2. TOO_NARROW: Needs to be generalized
3. APPROPRIATE: Good granularity for research

Format your response as a JSON object with topics as keys and classifications as values.

Example: {{"topic1": "TOO_BROAD", "topic2": "APPROPRIATE"}}

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
    response = query_llm(model_name, prompt)
    evaluation = extract_json(response)
    
    if not evaluation or not isinstance(evaluation, dict):
        logging.warning("Failed to evaluate topics, marking all as APPROPRIATE")
        return {topic: "APPROPRIATE" for topic in topics}
    
    # Ensure all topics have an evaluation
    for topic in topics:
        if topic not in evaluation:
            evaluation[topic] = "APPROPRIATE"
    
    return evaluation

def refine_topics(model_name, topics, evaluation, user_query):
    refined_topics = []
    
    for topic, status in evaluation.items():
        if status == "TOO_BROAD":
            prompt = f"""
The topic '{topic}' is too general for effective research on the query:
{user_query}

Please break this down into 2-4 more specific subtopics that would be more useful for researching this query.

Format your response as a JSON list of strings.

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
            response = query_llm(model_name, prompt)
            subtopics = extract_json(response)
            
            if subtopics and isinstance(subtopics, list):
                refined_topics.extend(subtopics)
            else:
                refined_topics.append(topic)
                
        elif status == "TOO_NARROW":
            prompt = f"""
The topic '{topic}' is too specific for finding general information. Suggest 1 broader topic that would encompass this specific topic while still being relevant to:
{user_query}

Format your response as a JSON list of strings.

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
            response = query_llm(model_name, prompt)
            broader_topics = extract_json(response)
            
            if broader_topics and isinstance(broader_topics, list):
                refined_topics.extend(broader_topics)
            else:
                refined_topics.append(topic)
        else:
            refined_topics.append(topic)
    
    # Remove duplicates while preserving order
    seen = set()
    return [x for x in refined_topics if not (x in seen or seen.add(x))]

def verify_topic_relevance(model_name, topics, user_query):
    prompt = f"""
Given the original query:

"{user_query}"

Evaluate how relevant each of these topics is:

{json.dumps(topics)}

Rate each topic's relevance on a scale of 1-10. Format your response as a JSON object with topics as keys and scores as values.

Example: {{"topic1": 7, "topic2": 3}}

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
    response = query_llm(model_name, prompt)
    relevance_scores = extract_json(response)
    
    if not relevance_scores or not isinstance(relevance_scores, dict):
        logging.warning("Failed to get relevance scores, using default score of 8")
        return {topic: 8 for topic in topics}
    
    # Ensure all topics have a score
    for topic in topics:
        if topic not in relevance_scores:
            relevance_scores[topic] = 8
    
    return relevance_scores

def search_wikipedia_with_context(topic, context_query, num_sentences=8):
    try:
        search_results = wikipedia.search(topic)
        
        if not search_results:
            combined_search = f"{topic} {context_query}"
            search_results = wikipedia.search(combined_search)
            
        if not search_results:
            return f"No Wikipedia results found for '{topic}'."
            
        scored_results = []
        for result in search_results[:5]:
            topic_words = set(topic.lower().split())
            context_words = set(context_query.lower().split())
            
            topic_score = sum(1 for word in topic_words if word in result.lower())
            context_score = sum(1 for word in context_words if word in result.lower())
            
            scored_results.append((result, topic_score * 2 + context_score))
            
        best_result = max(scored_results, key=lambda x: x[1])[0]
        
        try:
            page_summary = wikipedia.summary(best_result, sentences=num_sentences)
            return f"Information about '{best_result}':\n\n{page_summary}"
        except wikipedia.DisambiguationError as e:
            # Handle disambiguation pages by selecting the first option
            if e.options:
                try:
                    page_summary = wikipedia.summary(e.options[0], sentences=num_sentences)
                    return f"Information about '{e.options[0]}':\n\n{page_summary}"
                except Exception as inner_e:
                    return f"Error retrieving information about '{best_result}': {str(inner_e)}"
            else:
                return f"Multiple options found for '{best_result}', but none could be retrieved."
        except Exception as e:
            return f"Error retrieving information about '{best_result}': {str(e)}"
    except Exception as e:
        logging.error(f"Error in Wikipedia search for '{topic}': {str(e)}")
        return f"Error searching Wikipedia for '{topic}': {str(e)}"

def gather_information(topics, user_query):
    topic_information = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_topic = {executor.submit(search_wikipedia_with_context, topic, user_query): topic 
                         for topic in topics}
        
        for future in concurrent.futures.as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                topic_information[topic] = future.result()
            except Exception as e:
                logging.error(f"Error gathering information for '{topic}': {str(e)}")
                topic_information[topic] = f"Failed to gather information: {str(e)}"
    
    return topic_information

def synthesize_information(model_name, topic_information, user_query):
    info_text = "\n\n".join([f"TOPIC: {topic}\n{info}" for topic, info in topic_information.items()])
    
    prompt = f"""
I've gathered information about several topics related to the query:
{user_query}

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
{user_query}

Your answer should be well-structured, balanced, and address all aspects of the query. Include specific examples where appropriate.
"""
    
    return query_llm(model_name, prompt)

def verify_response(model_name, generated_answer, user_query):
    prompt = f"""
Please verify this answer to the query:
{user_query}

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
    
    if verification and "VERIFICATION: The answer is accurate and complete" in verification:
        return generated_answer
    else:
        prompt = f"""
Based on this verification feedback:
{verification}

Please improve the following answer to the query:
{user_query}

ORIGINAL ANSWER:
{generated_answer}

Provide a revised answer that addresses the issues identified in the verification.
"""
        
        improved_response = query_llm(model_name, prompt)
        return improved_response if improved_response else generated_answer

def process_query(model_name, user_query):
    logging.info(f"Processing query: '{user_query}'")
    
    topics = extract_topics(model_name, user_query)
    logging.info(f"Extracted topics: {topics}")
    
    evaluation = evaluate_topic_granularity(model_name, topics, user_query)
    logging.info(f"Topic evaluation: {evaluation}")
    
    refined_topics = refine_topics(model_name, topics, evaluation, user_query)
    logging.info(f"Refined topics: {refined_topics}")
    
    relevance_scores = verify_topic_relevance(model_name, refined_topics, user_query)
    logging.info(f"Topic relevance scores: {relevance_scores}")
    
    final_topics = [topic for topic in refined_topics if relevance_scores.get(topic, 0) >= 6]
    if not final_topics:
        logging.warning("No topics passed relevance threshold, using all topics")
        final_topics = refined_topics
    logging.info(f"Final topics after relevance filtering: {final_topics}")
    
    topic_information = gather_information(final_topics, user_query)
    logging.info(f"Gathered information for {len(topic_information)} topics")
    
    synthesized_info = synthesize_information(model_name, topic_information, user_query)
    logging.info("Information synthesized")
    
    response = generate_response(model_name, synthesized_info, user_query)
    logging.info("Response generated")
    
    verified_response = verify_response(model_name, response, user_query)
    logging.info("Response verified")
    
    logging.info("Query processing complete")
    return verified_response

def main():
    parser = argparse.ArgumentParser(description='Advanced Wikipedia-Ollama Query System')
    parser.add_argument('--model', default='gemma3:12b', help='Ollama model to use')
    parser.add_argument('--log-file', default='advanced_wiki_ollama.log', help='Log file path')
    parser.add_argument('--console-log', action='store_true', help='Also log to console')
    args = parser.parse_args()
    
    logger = setup_logging(log_to_console=args.console_log, log_file=args.log_file)
    
    logging.info(f"Starting advanced query system with Ollama model: {args.model}")
    print(f"Using Ollama model: {args.model}")
    print(f"Logging to file: {os.path.abspath(args.log_file)}")
    print("Enter your query (or 'exit' to quit):")
    
    while True:
        try:
            user_input = input("> ")
            
            if user_input.lower() in ['exit', 'quit']:
                logging.info("User requested to exit the application")
                break
            
            if not user_input.strip():
                print("Please enter a valid query.")
                continue
            
            print("\nProcessing your query. This may take a few moments...\n")
            
            start_time = time.time()
            response = process_query(args.model, user_input)
            end_time = time.time()
            
            if response:
                print("\nAnswer:")
                print(response)
                print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
            else:
                print("\nFailed to generate a response. Please check the logs for details.")
                
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()