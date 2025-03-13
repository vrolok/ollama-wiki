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
from functools import lru_cache
from bs4 import GuessedAtParserWarning
import warnings

# Suppress the BeautifulSoup warning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

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

def query_llm(model_name, prompt, max_retries=3, retry_delay=2, operation_name="LLM Query", timeout=60):
    for attempt in range(max_retries):
        try:
            logging.info(f"Starting {operation_name}")
            start_time = time.time()
            
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            elapsed = time.time() - start_time
            logging.info(f"Completed {operation_name} in {elapsed:.2f}s")
            return response['message']['content']
        except Exception as e:
            logging.error(f"Error in {operation_name} (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None

def extract_topics(model_name, user_query):
    prompt = f"""
I need you to analyze this query and extract key topics that would be ideal for Wikipedia research:
{user_query}

Format your response as a JSON list of strings containing only the key topics.
Example: ["topic1", "topic2", "topic3"]

IMPORTANT: 
1. Return ONLY valid JSON with no additional text!
2. Limit to 5-7 most important topics.
3. Keep each topic name under 50 characters.
4. Choose topics that are likely to have dedicated Wikipedia articles.
5. Prefer established concepts, technologies, or methodologies rather than specific implementations.
6. Avoid overly technical compound phrases that wouldn't appear as Wikipedia article titles.
"""
    response = query_llm(model_name, prompt, operation_name="Topic Extraction")
    topics = extract_json(response)
    
    if not topics or not isinstance(topics, list):
        logging.warning("Failed to extract topics, using fallback topics")
        return ['Error Handling', 'API Design', 'REST API Development']
    
    # Ensure topics are not too long
    topics = [topic[:50] for topic in topics]
    
    return topics

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

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
    response = query_llm(model_name, prompt, operation_name="Topic Granularity Evaluation")
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
"{user_query}"

Please break this down into 2-3 more specific subtopics that would be more useful for researching this query.
Format your response as a JSON list of strings.

IMPORTANT: 
1. Return ONLY valid JSON with no additional text!
2. Keep each topic name under 50 characters.
3. Choose topics that are likely to have dedicated Wikipedia articles.
"""
            response = query_llm(model_name, prompt, operation_name=f"Refining Broad Topic: {topic}")
            subtopics = extract_json(response)
            
            if subtopics and isinstance(subtopics, list):
                # Ensure subtopics are not too long
                subtopics = [subtopic[:50] for subtopic in subtopics]
                refined_topics.extend(subtopics)
            else:
                refined_topics.append(topic)
                
        elif status == "TOO_NARROW":
            prompt = f"""
The topic '{topic}' is too specific for finding general information.
Please suggest 1-2 broader topics that would encompass this specific topic while still being relevant to:
"{user_query}"

Format your response as a JSON list of strings.

IMPORTANT: 
1. Return ONLY valid JSON with no additional text!
2. Keep each topic name under 50 characters.
3. Choose topics that are likely to have dedicated Wikipedia articles.
"""
            response = query_llm(model_name, prompt, operation_name=f"Refining Narrow Topic: {topic}")
            broader_topics = extract_json(response)
            
            if broader_topics and isinstance(broader_topics, list):
                # Ensure topics are not too long
                broader_topics = [topic[:50] for topic in broader_topics]
                refined_topics.extend(broader_topics)
            else:
                refined_topics.append(topic)
        else:
            refined_topics.append(topic)
    
    # Remove duplicates while preserving order
    seen = set()
    refined_topics = [x for x in refined_topics if not (x in seen or seen.add(x))]
    
    # Limit to maximum 10 topics
    return refined_topics[:10]

def optimize_topics_for_wikipedia(model_name, topics):
    prompt = f"""
I have the following topics that need to be researched on Wikipedia:
{json.dumps(topics)}

For each topic, suggest a modified version that would be more likely to match actual Wikipedia article titles.
For example:
- "Microservices architecture patterns" might become "Microservices"
- "HTTP status codes for REST APIs" might become "HTTP status codes"
- "Node.js error handling best practices" might become "Node.js" and "Error handling"

Format your response as a JSON object with original topics as keys and optimized topics as values.
Example: {{"original topic": "optimized topic"}}

IMPORTANT: 
1. Return ONLY valid JSON with no additional text!
2. Keep each optimized topic under 50 characters.
3. Focus on terms that would likely be Wikipedia article titles.
"""
    response = query_llm(model_name, prompt, operation_name="Wikipedia Topic Optimization")
    optimized_topics = extract_json(response)
    
    if not optimized_topics or not isinstance(optimized_topics, dict):
        logging.warning("Failed to optimize topics for Wikipedia, using original topics")
        return topics
    
    # Extract the optimized topics and ensure no duplicates
    wiki_friendly_topics = []
    seen = set()
    
    for original, optimized in optimized_topics.items():
        if optimized and optimized not in seen:
            wiki_friendly_topics.append(optimized[:50])
            seen.add(optimized)
        elif original not in seen:
            wiki_friendly_topics.append(original[:50])
            seen.add(original)
    
    return wiki_friendly_topics

def verify_topic_relevance(model_name, topics, user_query):
    prompt = f"""
Given the original query:
"{user_query}"

Evaluate how relevant each of these topics is:
{json.dumps(topics)}

Rate each topic's relevance on a scale of 1-10.
Format your response as a JSON object with topics as keys and scores as values.
Example: {{"topic1": 8, "topic2": 5}}

IMPORTANT: Return ONLY valid JSON with no additional text.
"""
    response = query_llm(model_name, prompt, operation_name="Topic Relevance Scoring")
    relevance_scores = extract_json(response)
    
    if not relevance_scores or not isinstance(relevance_scores, dict):
        logging.warning("Failed to get relevance scores, using default score of 8")
        return {topic: 8 for topic in topics}
    
    # Ensure all topics have a score
    for topic in topics:
        if topic not in relevance_scores:
            relevance_scores[topic] = 8
    
    return relevance_scores

@lru_cache(maxsize=100)
def search_wikipedia_cached(search_term, num_sentences=8):
    """Cached version of Wikipedia search to avoid duplicate searches"""
    try:
        search_results = wikipedia.search(search_term)
        if not search_results:
            return f"No Wikipedia results found for '{search_term}'."
            
        best_result = search_results[0]
        
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
        logging.error(f"Error in Wikipedia search for '{search_term}': {str(e)}")
        return f"Error searching Wikipedia for '{search_term}': {str(e)}"

def search_wikipedia_with_context(topic, context_query, num_sentences=8):
    # Ensure topic is not too long for Wikipedia API (limit to 250 chars to be safe)
    if len(topic) > 250:
        logging.warning(f"Topic too long for Wikipedia search: '{topic}'. Truncating.")
        topic = topic[:250]
    
    try:
        # First try the topic directly
        result = search_wikipedia_cached(topic, num_sentences)
        
        # If no results, try with context
        if "No Wikipedia results found" in result:
            # Create a shorter combined search term
            keywords = " ".join(topic.split()[:3] + context_query.split()[:3])
            result = search_wikipedia_cached(keywords, num_sentences)
            
        return result
    except Exception as e:
        logging.error(f"Error in Wikipedia search for '{topic}': {str(e)}")
        return f"Error searching Wikipedia for '{topic}': {str(e)}"

def gather_information(topics, user_query):
    topic_information = {}
    
    # Use rate limiting to avoid overwhelming the Wikipedia API
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_topic = {executor.submit(search_wikipedia_with_context, topic, user_query): topic 
                         for topic in topics}
        
        for future in concurrent.futures.as_completed(future_to_topic):
            topic = future_to_topic[future]
            try:
                topic_information[topic] = future.result()
                # Add a small delay between requests
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Error gathering information for '{topic}': {str(e)}")
                topic_information[topic] = f"Failed to gather information: {str(e)}"
    
    return topic_information

def synthesize_information(model_name, topic_information, user_query):
    info_text = "\n\n".join([f"TOPIC: {topic}\n{info}" for topic, info in topic_information.items()])
    
    prompt = f"""
I've gathered information about several topics related to the query:
"{user_query}"

Please synthesize this information into a coherent knowledge base that addresses all aspects of the query.

{info_text}

Your synthesis should organize the information logically, identify connections between topics, and highlight the most relevant points for answering the query.
"""
    
    return query_llm(model_name, prompt, operation_name="Information Synthesis")

def generate_response(model_name, synthesized_info, user_query):
    prompt = f"""
Based on this synthesized information:

{synthesized_info}

Please provide a comprehensive answer to the original query:
"{user_query}"

Your answer should be well-structured, balanced, and address all aspects of the query. Include specific examples where appropriate.
"""
    
    return query_llm(model_name, prompt, operation_name="Response Generation")

def convert_markdown_to_plaintext(markdown_text):
    """Convert markdown formatting to plain text"""
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', markdown_text)
    
    # Remove inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Convert headers
    text = re.sub(r'^#{1,6}\s+(.+)', r'\1', text, flags=re.MULTILINE)
    
    # Convert bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Convert bullet points
    text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
    
    # Convert numbered lists
    text = re.sub(r'^\s*\d+\.\s+', lambda m: f"{m.group().strip()} ", text, flags=re.MULTILINE)
    
    # Remove link formatting but keep the text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    return text

def verify_response(model_name, generated_answer, user_query, plain_text=False):
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
    
    verification = query_llm(model_name, prompt, operation_name="Response Verification")
    
    if verification and "VERIFICATION: The answer is accurate and complete" in verification:
        result = generated_answer
    else:
        prompt = f"""
Based on this verification feedback:

{verification}

Please improve the following answer to the query:
"{user_query}"

ORIGINAL ANSWER:
{generated_answer}

Provide a revised answer that addresses the issues identified in the verification.
"""
        
        improved_response = query_llm(model_name, prompt, operation_name="Response Improvement")
        result = improved_response if improved_response else generated_answer
    
    if plain_text:
        # Convert markdown to plain text
        result = convert_markdown_to_plaintext(result)
    
    return result

def process_query(model_name, user_query, plain_text=False, max_topics=7):
    logging.info(f"Processing query: '{user_query}'")
    
    # Report progress to user
    print("Extracting key topics...")
    topics = extract_topics(model_name, user_query)
    logging.info(f"Extracted topics: {topics}")
    
    print("Evaluating topic granularity...")
    evaluation = evaluate_topic_granularity(model_name, topics, user_query)
    logging.info(f"Topic evaluation: {evaluation}")
    
    print("Refining topics...")
    refined_topics = refine_topics(model_name, topics, evaluation, user_query)
    logging.info(f"Refined topics: {refined_topics}")
    
    print("Optimizing topics for Wikipedia search...")
    wiki_topics = optimize_topics_for_wikipedia(model_name, refined_topics)
    logging.info(f"Wikipedia-optimized topics: {wiki_topics}")
    
    print("Scoring topic relevance...")
    relevance_scores = verify_topic_relevance(model_name, wiki_topics, user_query)
    logging.info(f"Topic relevance scores: {relevance_scores}")
    
    # Filter to most relevant topics (score >= 7)
    final_topics = [topic for topic in wiki_topics if relevance_scores.get(topic, 0) >= 7]
    if len(final_topics) < 3:
        # Ensure we have at least 3 topics
        remaining = [t for t in wiki_topics if t not in final_topics]
        remaining.sort(key=lambda t: relevance_scores.get(t, 0), reverse=True)
        final_topics.extend(remaining[:3-len(final_topics)])
    
    # Cap at maximum topics
    final_topics = final_topics[:max_topics]
    logging.info(f"Final topics after relevance filtering: {final_topics}")
    
    print(f"Gathering information on {len(final_topics)} topics...")
    topic_information = gather_information(final_topics, user_query)
    logging.info(f"Gathered information for {len(topic_information)} topics")
    
    print("Synthesizing information...")
    synthesized_info = synthesize_information(model_name, topic_information, user_query)
    logging.info("Information synthesized")
    
    print("Generating response...")
    response = generate_response(model_name, synthesized_info, user_query)
    logging.info("Response generated")
    
    print("Verifying response quality...")
    verified_response = verify_response(model_name, response, user_query, plain_text=plain_text)
    logging.info("Response verified")
    
    logging.info("Query processing complete")
    return verified_response

def main():
    parser = argparse.ArgumentParser(description='Advanced Wikipedia-Ollama Query System')
    parser.add_argument('--model', default='gemma3:12b', help='Ollama model to use')
    parser.add_argument('--log-file', default='advanced_wiki_ollama.log', help='Log file path')
    parser.add_argument('--console-log', action='store_true', help='Also log to console')
    parser.add_argument('--max-topics', type=int, default=7, help='Maximum number of topics to process')
    parser.add_argument('--plain-text', action='store_true', help='Output plain text instead of markdown')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for LLM requests')
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
            response = process_query(
                args.model, 
                user_input, 
                plain_text=args.plain_text,
                max_topics=args.max_topics
            )
            end_time = time.time()
            
            if response:
                print("\nAnswer:")
                print(response)
                print(f"\nProcessing time: {end_time - start_time:.2f} seconds")
            else:
                print("\nFailed to generate a response. Please check the logs for details.")
                
            print("\n" + "-"*50 + "\n")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            logging.info("Operation cancelled by user (KeyboardInterrupt)")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            logging.error(f"Unhandled exception in main loop: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()