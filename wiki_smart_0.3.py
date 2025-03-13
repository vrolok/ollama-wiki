import ollama
import wikipedia
import logging
import os
import json
import time
from logging.handlers import RotatingFileHandler
import re
import argparse
from functools import lru_cache
from bs4 import GuessedAtParserWarning
import warnings
import tiktoken

# Suppress the BeautifulSoup warning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# ================ Logging Configuration ================

class LoggingManager:
    @staticmethod
    def setup_logging(log_to_console=True, log_file="wiki_ollama.log", debug=False):
        logger = logging.getLogger()
        level = logging.DEBUG if debug else logging.INFO
        logger.setLevel(level)
        logger.handlers = []  # Clear existing handlers

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')

        # File handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

# ================ LLM Interaction ================

class LLMClient:
    def __init__(self, model_name):
        self.model_name = model_name
        # Initialize tiktoken encoder - use cl100k_base for most modern models
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logging.warning(f"Fallback to p50k_base encoder: {str(e)}")
            # Fallback to p50k_base if cl100k not available
            self.encoder = tiktoken.get_encoding("p50k_base")

    def count_tokens(self, text):
        """Count tokens for the given text using tiktoken"""
        try:
            tokens = self.encoder.encode(text)
            return len(tokens)
        except Exception as e:
            logging.warning(f"Error counting tokens: {str(e)}")
            # Fallback approximation (4 chars per token is a rough estimate)
            return len(text) // 4

    def fit_to_context_window(self, text, max_tokens, reserve_tokens=0):
        """Ensure text fits within the context window"""
        if not text:
            return ""

        total_tokens = self.count_tokens(text)
        if total_tokens <= max_tokens - reserve_tokens:
            return text

        # Need to truncate
        try:
            tokens = self.encoder.encode(text)
            truncated_tokens = tokens[:max_tokens-reserve_tokens]
            truncated = self.encoder.decode(truncated_tokens)
            return truncated
        except Exception as e:
            logging.error(f"Error truncating text: {str(e)}")
            # Fallback to rough approximation
            ratio = (max_tokens - reserve_tokens) / total_tokens
            char_limit = int(len(text) * ratio)
            return text[:char_limit]

    def chunk_content(self, content, max_tokens_per_chunk=8000):
        """Split content into chunks that respect token limits"""
        if not content:
            return []

        try:
            tokens = self.encoder.encode(content)
            chunks = []

            for i in range(0, len(tokens), max_tokens_per_chunk):
                chunk_tokens = tokens[i:i+max_tokens_per_chunk]
                chunk_text = self.encoder.decode(chunk_tokens)
                chunks.append(chunk_text)

            return chunks
        except Exception as e:
            logging.error(f"Error chunking with tiktoken: {str(e)}")
            # Fall back to paragraph-based chunking
            return self._chunk_by_paragraphs(content, max_tokens_per_chunk)

    def _chunk_by_paragraphs(self, content, max_tokens_per_chunk):
        """Fallback chunking by paragraphs"""
        paragraphs = content.split("\n\n")
        chunks = []
        current_chunk = []
        current_token_estimate = 0

        for paragraph in paragraphs:
            # Rough token estimation
            paragraph_tokens = len(paragraph.split()) * 1.3

            if current_token_estimate + paragraph_tokens > max_tokens_per_chunk and current_chunk:
                # This chunk would be too large, finalize the current one
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_token_estimate = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_token_estimate += paragraph_tokens

        # Add the last chunk if it has content
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def query(self, prompt, max_retries=3, retry_delay=2, operation_name="LLM Query", timeout=60):
        for attempt in range(max_retries):
            try:
                logging.info(f"Starting {operation_name}")
                logging.debug(f"PROMPT for {operation_name}:\n{prompt}")

                start_time = time.time()

                # Check prompt token count
                prompt_tokens = self.count_tokens(prompt)
                logging.info(f"{operation_name} - Prompt token count: {prompt_tokens}")

                response = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}]
                )

                content = response['message']['content']
                elapsed = time.time() - start_time
                logging.info(f"Completed {operation_name} in {elapsed:.2f}s")
                logging.debug(f"RESPONSE for {operation_name}:\n{content}")

                return content
            except Exception as e:
                logging.error(f"Error in {operation_name} (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return None

# ================ JSON Processing ================

class JSONProcessor:
    @staticmethod
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

# ================ Wikipedia Interaction ================

class WikipediaClient:
    def __init__(self):
        # Don't use Wikipedia's rate limiting as it has issues
        # Instead we'll manually control timing between requests
        pass

    @staticmethod
    @lru_cache(maxsize=100)
    def search_wikipedia(search_term):
        """Basic Wikipedia search returning the best match"""
        try:
            logging.debug(f"Searching Wikipedia for: '{search_term}'")
            search_results = wikipedia.search(search_term, results=5)
            time.sleep(1)  # Manual rate limiting

            if not search_results:
                logging.debug(f"No Wikipedia results for: '{search_term}'")
                return None

            logging.debug(f"Wikipedia search results for '{search_term}': {search_results[:3]}")
            return search_results[0]
        except Exception as e:
            logging.error(f"Error in Wikipedia search for '{search_term}': {str(e)}")
            return None

    @staticmethod
    def get_page_content(page_title):
        """Get full Wikipedia page content"""
        try:
            logging.debug(f"Retrieving Wikipedia page: '{page_title}'")

            # First try to get just the summary
            summary = wikipedia.summary(page_title, sentences=1, auto_suggest=False)
            time.sleep(1)  # Manual rate limiting

            # If summary is available, get the full page
            page = wikipedia.page(page_title, auto_suggest=False)
            time.sleep(1)  # Manual rate limiting

            result = {
                'title': page.title,
                'content': page.content,
                'url': page.url,
                'summary': summary
            }
            logging.debug(f"Successfully retrieved page '{page_title}' ({len(page.content)} chars)")
            return result

        except wikipedia.DisambiguationError as e:
            if e.options and len(e.options) > 0:
                logging.debug(f"Disambiguation page for '{page_title}', trying first option: {e.options[0]}")
                time.sleep(1)  # Wait before trying alternative
                try:
                    return WikipediaClient.get_page_content(e.options[0])
                except Exception as inner_e:
                    logging.error(f"Error retrieving page content for disambiguation option: {str(inner_e)}")
                    return {'error': str(inner_e), 'title': page_title}
            return {'error': 'Disambiguation error with no options', 'title': page_title}
        except Exception as e:
            logging.error(f"Error retrieving page content for '{page_title}': {str(e)}")
            return {'error': str(e), 'title': page_title}

    def retrieve_topic_information(self, topic, user_query):
        """Retrieve relevant Wikipedia information for a topic with better error handling"""
        try:
            logging.debug(f"Starting information retrieval for topic: '{topic}'")

            # First try the exact topic
            best_match = self.search_wikipedia(topic)

            # If no match, try with context
            if not best_match:
                logging.debug(f"No exact match for '{topic}', trying simplified keywords")
                keywords = " ".join(topic.split()[:3])
                best_match = self.search_wikipedia(keywords)

                # If still no match, try with query context
                if not best_match:
                    logging.debug(f"No match for keywords '{keywords}', trying with query context")
                    combined_search = " ".join(topic.split()[:2] + user_query.split()[:2])
                    best_match = self.search_wikipedia(combined_search)

                    if not best_match:
                        logging.debug(f"All search attempts failed for '{topic}'")
                        return {
                            "error": f"No Wikipedia results found for '{topic}'",
                            "title": topic,
                            "content": f"No Wikipedia results found for '{topic}'."
                        }

            logging.debug(f"Best Wikipedia match for '{topic}': '{best_match}'")

            # Get the page content with manual delay to avoid rate limiting issues
            time.sleep(1)
            page_data = self.get_page_content(best_match)

            # If there's an error but we have a title, return basic information
            if page_data.get('error') and page_data.get('title'):
                logging.debug(f"Error retrieving content for '{best_match}': {page_data.get('error')}")
                return {
                    "error": page_data.get('error'),
                    "title": page_data.get('title'),
                    "content": f"Error retrieving content for '{page_data.get('title')}': {page_data.get('error')}"
                }

            logging.debug(f"Successfully retrieved information for topic '{topic}' via '{best_match}'")
            return page_data

        except Exception as e:
            logging.error(f"Unexpected error retrieving information for '{topic}': {str(e)}")
            return {
                "error": str(e),
                "title": topic,
                "content": f"Failed to retrieve information for '{topic}': {str(e)}"
            }

# ================ Topic Processing ================

class TopicProcessor:
    def __init__(self, llm_client, json_processor):
        self.llm = llm_client
        self.json = json_processor

    def extract_topics(self, user_query):
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
        """
        response = self.llm.query(prompt, operation_name="Topic Extraction")
        topics = self.json.extract_json(response)

        if not topics or not isinstance(topics, list):
            logging.warning("Failed to extract topics, using fallback topics")
            return ['Error Handling', 'API Design', 'REST API Development']

        # Ensure topics are not too long
        topics = [topic[:50] for topic in topics]
        return topics

    def optimize_topics_for_wikipedia(self, topics):
        prompt = f"""
I have the following topics that need to be researched on Wikipedia:
{json.dumps(topics)}

For each topic, suggest a modified version that would be more likely to match actual Wikipedia article titles.
Format your response as a JSON object with original topics as keys and optimized topics as values.
Example: {{"original topic": "optimized topic"}}

IMPORTANT:
1. Return ONLY valid JSON with no additional text!
2. Keep each optimized topic under 50 characters.
3. Focus on terms that would likely be Wikipedia article titles.
        """
        response = self.llm.query(prompt, operation_name="Wikipedia Topic Optimization")
        optimized_topics = self.json.extract_json(response)

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

    def score_topic_relevance(self, topics, user_query):
        prompt = f"""
Given the original query: "{user_query}"

Evaluate how relevant each of these topics is:
{json.dumps(topics)}

Rate each topic's relevance on a scale of 1-10.
Format your response as a JSON object with topics as keys and scores as values.
Example: {{"topic1": 8, "topic2": 5}}

IMPORTANT: Return ONLY valid JSON with no additional text.
        """
        response = self.llm.query(prompt, operation_name="Topic Relevance Scoring")
        relevance_scores = self.json.extract_json(response)

        if not relevance_scores or not isinstance(relevance_scores, dict):
            logging.warning("Failed to get relevance scores, using default score of 8")
            return {topic: 8 for topic in topics}

        # Ensure all topics have a score
        for topic in topics:
            if topic not in relevance_scores:
                relevance_scores[topic] = 8

        return relevance_scores

# ================ Information Processing ================

class InformationProcessor:
    def __init__(self, llm_client, wiki_client):
        self.llm = llm_client
        self.wiki = wiki_client

    def extract_relevant_information(self, topic, content, user_query):
        """Extract query-relevant information from topic content with proper token management"""
        if not content:
            return f"No information available for {topic}."

        # Define token limits
        prompt_tokens = 1000  # Reserve for prompt template
        max_response_tokens = 2000  # Reserve for response
        model_context_limit = 32000  # Adjust based on your model

        # Calculate available tokens for content
        available_content_tokens = model_context_limit - prompt_tokens - max_response_tokens

        # Prepare content with proper token counting
        prepared_content = self.llm.fit_to_context_window(
            content,
            available_content_tokens
        )

        # If content is very large, process in chunks and combine results
        if len(content) > len(prepared_content) * 1.5:  # Content was significantly truncated
            logging.info(f"Content for '{topic}' exceeds token limit. Processing in chunks.")
            chunks = self.llm.chunk_content(content, max_tokens_per_chunk=available_content_tokens)

            if len(chunks) > 1:
                all_extractions = []
                for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks to avoid too many API calls
                    chunk_prompt = f"""
I have PART {i+1} of {len(chunks[:3])} about "{topic}" from Wikipedia.
Given the user query: "{user_query}"

Please analyze this content part and extract ONLY the information that is directly
relevant to answering this specific query. Focus on facts, data, and insights.

CONTENT PART {i+1}:
{chunk}

Return ONLY the relevant extracted information, organized by importance.
                    """
                    extraction = self.llm.query(
                        chunk_prompt,
                        operation_name=f"Information Extraction: {topic} (Part {i+1})"
                    )
                    if extraction:
                        all_extractions.append(extraction)

                # If we have multiple extractions, combine them
                if len(all_extractions) > 1:
                    combined_extractions = "\n\n".join([f"PART {i+1}:\n{ext}" for i, ext in enumerate(all_extractions)])

                    synthesis_prompt = f"""
I've extracted relevant information from multiple parts of content about "{topic}"
related to the query: "{user_query}"

Here are the extractions from different parts:

{combined_extractions}

Please synthesize these extractions into a single coherent summary of the
most relevant information from this topic for answering the query.
Eliminate redundancies and organize by importance.
                    """

                    return self.llm.query(
                        synthesis_prompt,
                        operation_name=f"Information Synthesis: {topic}"
                    )
                elif all_extractions:
                    return all_extractions[0]
                else:
                    return f"No relevant information found in '{topic}'."

        # Standard single-chunk processing
        prompt = f"""
I have content about "{topic}" from Wikipedia.
Given the user query: "{user_query}"

Please analyze this content and extract ONLY the information that is directly
relevant to answering this specific query. Focus on facts, data, and insights
that would help address the query.

CONTENT:
```
{prepared_content}
```
Return ONLY the relevant information to - "{user_query}", organized by importance.
        """
        result = self.llm.query(prompt, operation_name=f"Information Extraction: {topic}")
        return result or f"Failed to extract information from '{topic}'."

    def analyze_cross_topic_relationships(self, topic_extractions, user_query):
        """Analyze relationships between information from different topics"""
        if not topic_extractions:
            return "No topic information available to analyze."

        combined_extractions = "\n\n".join([f"TOPIC: {topic}\n{info}"
                                          for topic, info in topic_extractions.items()])

        # Check token count
        token_count = self.llm.count_tokens(combined_extractions)
        if token_count > 20000:  # If extractions are very large
            logging.info(f"Combined extractions exceed 20K tokens ({token_count}). Summarizing individual topics first.")

            # First summarize each extraction to reduce size
            summarized_extractions = {}
            for topic, extraction in topic_extractions.items():
                summarize_prompt = f"""
Summarize this information about "{topic}" in relation to the query: "{user_query}"
Focus on the most important points only, in 600 words or less.

{extraction}
                """
                summary = self.llm.query(summarize_prompt, operation_name=f"Summarizing extraction: {topic}")
                if summary:
                    summarized_extractions[topic] = summary

            # Update combined extractions with summarized versions
            combined_extractions = "\n\n".join([f"TOPIC: {topic}\n{info}"
                                              for topic, info in summarized_extractions.items()])

        prompt = f"""
I've extracted relevant information from multiple topics related to: "{user_query}"

```
{combined_extractions}
```

Please analyze above information and identify:
1. Key connections between different topics
2. Complementary information across topics
3. Any contradictions or inconsistencies
4. Which information is most central to answering the query - "{user_query}"

Provide a structured analysis of how these topics relate to each other in the context of the query.
        """
        return self.llm.query(prompt, operation_name="Cross-Topic Analysis")

    def synthesize_final_response(self, cross_topic_analysis, user_query):
        """Generate final comprehensive response"""
        if not cross_topic_analysis:
            return "Unable to analyze the information from the topics."

        prompt = f"""
The cross-topic analysis:

```
{cross_topic_analysis}
```

Provide a comprehensive answer to the original query: "{user_query}"

Your answer should be well-structured, balanced, and address all aspects of the query.
Include specific examples where appropriate and organize the information in a logical flow.
        """
        return self.llm.query(prompt, operation_name="Final Response Generation")

    def verify_response(self, response, user_query, plain_text=False):
        """Verify and improve the final response if needed"""
        if not response:
            return "Failed to generate a response to your query."

        prompt = f"""
Please verify this answer to the query:
"{user_query}"

ANSWER:
{response}

Check for:
1. Factual accuracy
2. Completeness (addresses all aspects of the query)
3. Logical coherence

If you find any issues, please provide an improved version of the answer.
If the answer is satisfactory, respond with "VERIFICATION: The answer is accurate and complete."
        """
        verification = self.llm.query(prompt, operation_name="Response Verification")

        if verification and "VERIFICATION: The answer is accurate and complete" in verification:
            result = response
        else:
            # Extract the improved answer if verification found issues
            improved_response = self.llm.query(
                f"""
Based on this verification feedback:
{verification}

Please provide an improved answer to:
"{user_query}"
                """,
                operation_name="Response Improvement"
            )
            result = improved_response if improved_response else response

        if plain_text:
            result = self.convert_markdown_to_plaintext(result)

        return result

    @staticmethod
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

# ================ Main Query Processor ================

class QueryProcessor:
    def __init__(self, model_name):
        self.llm_client = LLMClient(model_name)
        self.json_processor = JSONProcessor()
        self.wiki_client = WikipediaClient()
        self.topic_processor = TopicProcessor(self.llm_client, self.json_processor)
        self.info_processor = InformationProcessor(self.llm_client, self.wiki_client)

    def process_query(self, user_query, plain_text=False, max_topics=5):
        logging.info(f"Processing query: '{user_query}'")

        # Step 1: Extract and optimize topics
        print("Extracting key topics...")
        initial_topics = self.topic_processor.extract_topics(user_query)
        logging.info(f"Extracted topics: {initial_topics}")

        print("Optimizing topics for Wikipedia search...")
        wiki_topics = self.topic_processor.optimize_topics_for_wikipedia(initial_topics)
        logging.info(f"Wikipedia-optimized topics: {wiki_topics}")

        print("Scoring topic relevance...")
        relevance_scores = self.topic_processor.score_topic_relevance(wiki_topics, user_query)
        logging.info(f"Topic relevance scores: {relevance_scores}")

        # Filter to most relevant topics
        final_topics = sorted(wiki_topics, key=lambda t: relevance_scores.get(t, 0), reverse=True)[:max_topics]
        logging.info(f"Final topics after relevance filtering: {final_topics}")

        # Step 2: Gather information
        print(f"Gathering information on {len(final_topics)} topics...")
        topic_content = {}

        # Use serial processing with careful error handling and manual rate limiting
        for i, topic in enumerate(final_topics):
            try:
                print(f"  - Searching for information on: {topic} ({i+1}/{len(final_topics)})")
                result = self.wiki_client.retrieve_topic_information(topic, user_query)
                topic_content[topic] = result
                # Manual delay between topics
                if i < len(final_topics) - 1:  # Don't sleep after the last topic
                    time.sleep(1.5)  # Increased delay to avoid rate limiting issues
            except Exception as e:
                logging.error(f"Error gathering information for '{topic}': {str(e)}")
                topic_content[topic] = {"error": str(e), "title": topic}

        # Step 3: Extract relevant information from each topic
        print("Extracting query-relevant information...")
        topic_extractions = {}

        for topic, data in topic_content.items():
            if data.get("error"):
                logging.warning(f"Skipping information extraction for '{topic}' due to retrieval error")
                continue

            content = data.get('content', '')
            if not content:
                logging.warning(f"No content available for '{topic}'")
                continue

            print(f"  - Analyzing content for: {data.get('title', topic)}")
            extraction = self.info_processor.extract_relevant_information(
                data.get('title', topic),
                content,
                user_query
            )
            if extraction:
                topic_extractions[data.get('title', topic)] = extraction

        if not topic_extractions:
            return "I couldn't find any relevant information for your query. Please try rephrasing or using different search terms."

        # Step 4: Cross-topic analysis
        print("Analyzing relationships between topics...")
        cross_topic_analysis = self.info_processor.analyze_cross_topic_relationships(topic_extractions, user_query)
        logging.info("Cross-topic analysis complete")

        # Step 5: Final synthesis
        print("Synthesizing final response...")
        response = self.info_processor.synthesize_final_response(cross_topic_analysis, user_query)
        logging.info("Response synthesis complete")

        # Step 6: Verify and improve
        print("Verifying response quality...")
        verified_response = self.info_processor.verify_response(response, user_query, plain_text=plain_text)
        logging.info("Response verification complete")

        return verified_response

# ================ Main Application ================

def main():
    parser = argparse.ArgumentParser(description='Advanced Wikipedia-Ollama Query System')
    parser.add_argument('--model', default='gemma3:12b', help='Ollama model to use')
    parser.add_argument('--log-file', default='wiki_ollama.log', help='Log file path')
    parser.add_argument('--console-log', action='store_true', help='Also log to console')
    parser.add_argument('--max-topics', type=int, default=5, help='Maximum number of topics to process')
    parser.add_argument('--plain-text', action='store_true', help='Output plain text instead of markdown')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    logger = LoggingManager.setup_logging(
        log_to_console=args.console_log,
        log_file=args.log_file,
        debug=args.debug
    )

    logging.info(f"Starting advanced query system with Ollama model: {args.model}")
    print(f"Using Ollama model: {args.model}")
    print(f"Logging to file: {os.path.abspath(args.log_file)}")
    if args.debug:
        print("DEBUG logging enabled - all prompts and responses will be logged")
    print("Enter your query (or 'exit' to quit):")

    processor = QueryProcessor(args.model)

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
            response = processor.process_query(
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
