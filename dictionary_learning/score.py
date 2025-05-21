import os
import json
import time
import asyncio
import aiohttp
import torch as t
import pandas as pd
from tqdm.asyncio import tqdm as async_tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import tqdm

from demo_config import LLM_CONFIG

OPENAI_API_KEY = ""

complexity_prompt = """
As an expert linguist and cognitive scientist specializing in text complexity analysis, your task is to perform a comprehensive evaluation of the provided text. You will assess its complexity on a scale from 0-10 (where 0 is extremely simple and 10 is highly complex) by analyzing multiple dimensions that contribute to cognitive processing difficulty.

## Detailed Evaluation Dimensions

### 1. Lexical Complexity (Weight: 20%)
Evaluate the vocabulary sophistication level using these criteria:
- **Word Frequency**: Proportion of uncommon words (not in the 5000 most frequent words)
- **Word Length**: Average syllable count and character length of words
- **Lexical Diversity**: Type-token ratio (unique words divided by total words)
- **Technical Terminology**: Presence of specialized or domain-specific vocabulary
- **Lexical Density**: Ratio of content words (nouns, verbs, adjectives, adverbs) to function words (pronouns, prepositions, articles, etc.)

### 2. Syntactic Complexity (Weight: 20%)
Analyze sentence structure complexity using these metrics:
- **Sentence Length**: Average number of words per sentence
- **Clause Density**: Number of clauses per sentence
- **Subordination**: Frequency and depth of subordinate clauses
- **Passive Voice**: Proportion of sentences in passive voice
- **Syntactic Variety**: Diversity of sentence structures
- **Embedding Depth**: How deeply clauses are nested within one another

### 3. Conceptual Density (Weight: 25%)
Assess the density and abstraction level of ideas presented:
- **Concept Count**: Number of distinct concepts, ideas, or arguments introduced
- **Concept Abstraction**: Level of concreteness vs. abstraction of concepts
- **Conceptual Networks**: Complexity of relationships between concepts
- **Information Density**: Amount of information conveyed per paragraph
- **Theoretical Complexity**: Depth of theoretical constructs presented

### 4. Domain Specificity (Weight: 15%)
Evaluate how much specialized domain knowledge is required:
- **Background Knowledge**: Prerequisite knowledge assumed by the text
- **Domain Vocabulary**: Concentration of field-specific terminology
- **Conceptual Familiarity**: How familiar concepts would be to general readers
- **Specialized References**: References to domain-specific methods, theories, or figures
- **Audience Specificity**: How targeted the text is to specialists vs. general readers

### 5. Logical Structure (Weight: 10%)
Analyze the complexity of reasoning patterns:
- **Argument Structure**: Complexity of argumentative or explanatory structure
- **Logical Operations**: Presence of conditional, causal, comparative reasoning
- **Inference Requirements**: How much the reader must infer rather than being told explicitly
- **Logical Connections**: Clarity and complexity of connections between ideas
- **Reasoning Chains**: Length and complexity of logical chains

### 6. Contextual Dependencies (Weight: 10%)
Assess how much the text relies on external context:
- **Intertextual References**: References to other texts or knowledge sources
- **Cultural Knowledge**: Required cultural or historical background
- **Implicit Information**: Amount of information that remains unstated but necessary
- **Presuppositions**: Assumptions the text makes about reader knowledge
- **Discourse Context**: How much meaning depends on broader discourse context

## Text to Evaluate:
{text}

## Required Output Format

ONLY return a JSON object with the following structure:
{{
  "lexical_complexity": {{
    "score": <0-10 number>
  }},
  "syntactic_complexity": {{
    "score": <0-10 number>
  }},
  "conceptual_density": {{
    "score": <0-10 number>
  }},
  "domain_specificity": {{
    "score": <0-10 number>
  }},
  "logical_structure": {{
    "score": <0-10 number>
  }},
  "contextual_dependencies": {{
    "score": <0-10 number>
  }},
  "final_weighted_score": <calculated final score as decimal>,
  "normalized_complexity_score": <rounded to one decimal place, e.g. 4.5>
}}
"""

class ContextsGenerator:
    
    def __init__(self, model_name, context_length=1024, max_contexts=None):
        self.model_name = model_name
        self.context_length = context_length
        self.max_contexts = max_contexts
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.llm_config = LLM_CONFIG[model_name]
        
        print(f"Initializing ContextsGenerator: model={model_name}, context_length={context_length}")
    
    def generate_contexts(self):
        dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        
        contexts = []
        token_buffer = []
        total_tokens = 0
        
        print("GPT-4.1-mini starting to generate contexts...")
        
        for item in tqdm.tqdm(dataset):
            text = item["text"]
            
            # Tokenize
            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)
            
            # Extract complete contexts when buffer is long enough
            while len(token_buffer) >= self.context_length:
                # Extract a complete context
                context_tokens = token_buffer[:self.context_length]
                token_buffer = token_buffer[self.context_length:]
                
                # Decode back to text
                context_text = self.tokenizer.decode(context_tokens)
                contexts.append(context_text)
                
                total_tokens += self.context_length
                
                # Check if maximum number of contexts reached
                if self.max_contexts and len(contexts) >= self.max_contexts:
                    print(f"Maximum context limit reached: {self.max_contexts}")
                    return contexts
                
                # Check if total token target reached
                if total_tokens >= 256_000_000:
                    print(f"Contexts of ~ 256,000,000 tokens have been generated")
                    return contexts
        
        return contexts


async def call_openai_api(session, api_key, text):
    if not text or text.strip() == "":
        return {
            "final_weighted_score": 0.0,
            "normalized_complexity_score": 0.0
        }
    
    request_data = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "You are an expert in text complexity analysis."},
            {"role": "user", "content": complexity_prompt.format(text=text)}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=request_data,
            timeout=30  # 30 seconds timeout
        ) as response:
            if response.status == 429:
                retry_after = int(response.headers.get("Retry-After", 10))
                print(f"Rate limit reached, waiting {retry_after} seconds")
                await asyncio.sleep(retry_after)
                return await call_openai_api(session, api_key, text)
            
            if response.status != 200:
                error_text = await response.text()
                print(f"API error status {response.status}: {error_text}")
                return {
                    "error": f"API returned status {response.status}",
                    "final_weighted_score": 0.0,
                    "normalized_complexity_score": 0.0
                }
                
            response_data = await response.json()
            
            if "error" in response_data:
                print(f"API error: {response_data['error']}")
                return {
                    "error": str(response_data['error']),
                    "final_weighted_score": 0.0,
                    "normalized_complexity_score": 0.0
                }
            
            content = response_data["choices"][0]["message"]["content"]
            try:
                result = json.loads(content)
                # Ensure normalized_complexity_score is one decimal place
                if "final_weighted_score" in result:
                    result["normalized_complexity_score"] = round(float(result["final_weighted_score"]), 1)
                return result
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}, content: {content[:100]}...")
                return {
                    "error": "Invalid JSON response",
                    "final_weighted_score": 0.0,
                    "normalized_complexity_score": 0.0
                }
            
    except asyncio.TimeoutError:
        print("Request timeout")
        return {
            "error": "Request timeout",
            "final_weighted_score": 0.0,
            "normalized_complexity_score": 0.0
        }
    except Exception as e:
        print(f"API call error: {str(e)}")
        return {
            "error": str(e),
            "final_weighted_score": 0.0,
            "normalized_complexity_score": 0.0
        }


async def process_batch(contexts, batch_indices, api_key, max_concurrency=5):
    async with aiohttp.ClientSession() as session:
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # Function to process a single request (with semaphore)
        async def process_with_semaphore(context, index):
            async with semaphore:
                # Add small delay between requests to avoid exceeding API limits
                await asyncio.sleep(0.1)
                result = await call_openai_api(session, api_key, context)
                # Display complexity score with one decimal place
                complexity = result.get('normalized_complexity_score', 0.0)
                print(f"Processed context {index}, result: complexity={complexity:.1f}")
                return result, index
        
        # Process all requests and show progress
        tasks = [process_with_semaphore(contexts[i], i) for i in batch_indices]
        results = await async_tqdm.gather(*tasks)
        
        return results


async def main_async(args):
    try:
        # Create context generator
        generator = ContextsGenerator(
            model_name=args.model_name,
            context_length=args.context_length,
            max_contexts=args.max_contexts
        )
        
        # Check if contexts have already been generated
        contexts_path = args.contexts_path
        if os.path.exists(contexts_path):
            print(f"Loading previously generated contexts from {contexts_path}")
            with open(contexts_path, "r") as f:
                contexts = json.load(f)
        else:
            # Generate contexts
            contexts = generator.generate_contexts()
            print(f"Generated {len(contexts)} contexts, approximately {len(contexts) * args.context_length} tokens")
            
            # Save generated contexts for future use
            with open(contexts_path, "w") as f:
                json.dump(contexts, f)
            print(f"Contexts saved to {contexts_path}")
        
        # Check for checkpoint
        checkpoint_path = args.output_path.replace(".parquet", "_checkpoint.parquet")
        start_index = 0
        all_scores = []
        
        if os.path.exists(checkpoint_path):
            print(f"Checkpoint found: {checkpoint_path}")
            checkpoint_df = pd.read_parquet(checkpoint_path)
            start_index = len(checkpoint_df)
            
            if start_index >= len(contexts):
                print("All contexts have been scored, saving final results directly")
                checkpoint_df.to_parquet(args.output_path)
                return
            
            all_scores = checkpoint_df['complexity_details'].tolist()
            print(f"Continuing from index {start_index} (completed {start_index}/{len(contexts)})")
        
        print(f"Will process remaining {len(contexts) - start_index} contexts starting from {start_index}")
        
        # Process in batches
        batch_size = args.batch_size
        max_concurrency = args.max_concurrency
        
        remaining_indices = list(range(start_index, len(contexts)))
        total_batches = (len(remaining_indices) + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(remaining_indices))
            batch_indices = remaining_indices[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_num + 1}/{total_batches} (contexts {batch_indices[0]}-{batch_indices[-1]})")
            
            results = await process_batch(
                contexts, 
                batch_indices, 
                api_key=OPENAI_API_KEY, 
                max_concurrency=max_concurrency
            )
            
            # Extend results and all_scores lists
            while len(all_scores) < batch_indices[-1] + 1:
                all_scores.append(None)
                
            for result, index in results:
                all_scores[index] = result
            
            # Create DataFrame up to current point
            df_contexts = contexts[:batch_indices[-1] + 1]
            normalized_scores = []
            
            for score in all_scores[:batch_indices[-1] + 1]:
                if isinstance(score, dict) and 'final_weighted_score' in score:
                    normalized_scores.append(round(float(score['final_weighted_score']), 1))
                else:
                    normalized_scores.append(0.0)  # Default value for error cases
            
            checkpoint_df = pd.DataFrame({
                'text': df_contexts,
                'complexity_score': normalized_scores,
                'complexity_details': all_scores[:batch_indices[-1] + 1]
            })
            
            # Save checkpoint
            checkpoint_df.to_parquet(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Brief pause between batches to avoid rate limits
            if batch_end < len(remaining_indices):
                pause_time = 1
                print(f"Pausing for {pause_time} seconds before next batch...")
                await asyncio.sleep(pause_time)
        
        # After all batches complete, save final results
        checkpoint_df.to_parquet(args.output_path)
        print(f"All context scoring complete! Final results saved to {args.output_path}")
        print(f"Scored a total of {len(checkpoint_df)} contexts, available for training")
        
    except Exception as e:
        print(f"Main async function error: {str(e)}")
        import traceback
        print(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Score language model training contexts for complexity using GPT-4.1-mini")
    
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m-deduped",
                       help="Name of language model to score, should match the one used for training")
    
    parser.add_argument("--context_length", type=int, default=1024,
                       help="Context length, should match the one used during training")
    
    parser.add_argument("--max_contexts", type=int, default=None,
                       help="Maximum number of contexts to score, None for unlimited")
    
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch processing size")
    
    parser.add_argument("--max_concurrency", type=int, default=5,
                       help="Maximum concurrent API requests")
    
    parser.add_argument("--contexts_path", type=str,
                       help="File path to save generated contexts")
    
    parser.add_argument("--output_path", type=str,
                       help="Output path for scored results parquet file")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main_async(args))
    except Exception as e:
        print(f"Main function error: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    main()