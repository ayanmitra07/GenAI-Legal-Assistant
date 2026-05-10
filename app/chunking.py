import re #Regular expressions for pattern matching
import json
from app.config import JSON_PATH

def create_chunks(full_text):
    """
    Takes full constitution text
    Returns structured chunks for retrieval
    """

    print("="*60)
    print("CHUNKING STARTED")
    print("="*60)

    # ============================================================
    # SECTION 2 — PART CHUNKING CONTD after ingestion.py (PRODUCTION STABLE)
    # ============================================================ 

    # ------------------------------------------------------------
    # STEP 4 — STRICT ARTICLE EXTRACTION
    # LLMs work best when context is structured. Instead of feeding Entire constitution we now have Article level retrieval
    # ------------------------------------------------------------

    print("Extracting articles...")
    #Pattern that identifies article headers. Once we know where each article starts, we can extract the text between them.
    #(?m) enables multiline mode, allowing the ^ anchor to match the start of each line. 
    # \s* matches any whitespace characters (like spaces or tabs) that may appear before the article number. 
    # (\d+[A-Z]?) captures the article number, which consists of one or more digits followed by an optional uppercase letter (to account for articles like 5A). 
    # \.\s+ matches a period followed by at least one space, which typically follows the article number in the text.
    #Start of line |optional spaces| article number| dot| space

    article_pattern = re.compile(   #re.compile() prepares a regex pattern once so it can be reused efficiently.
    r'(?m)^\s*(\d+[A-Z]?)\.\s+' #This regex pattern is designed to match the article numbers in the constitution text. It looks for lines that start with optional whitespace, followed by one or more digits, an optional uppercase letter, a period, and then at least one space. The (?m) at the beginning enables multiline mode, allowing the ^ anchor to match the start of each line.
    )

    matches = list(article_pattern.finditer(full_text)) #finditer() searches the text and returns all matches. Each match contains information about the article number and its position in the text. We convert this to a list for easier processing.

    #defensive programming: If the regex fails to detect articles, the pipeline stops.
    if not matches:
        raise Exception("No articles head detected. Extraction failed.")

    #Create storage Structure
    articles ={} #We will store the extracted articles in a dictionary where the key is the article number and the value is the article text.

    for i in range(len(matches)): #loops through every article header detected
        article_no = matches[i].group(1) #Extracts the article number from the regex match. group(1) refers to the first capturing group in the regex, which is (\d+[A-Z]?). This will give us the article number like "1", "2", "5A", etc.

        #Restrict numeric range defensively
        try:
            numeric_part = int(re.match(r'\d+', article_no).group()) #Extracts the numeric part of the article number. For example, if the article number is "5A", this will extract "5". This is done using another regex that looks for digits at the start of the article number.
        except: #If an error occurs in the try block, run this block instead
            continue #works inside loops, skip the current iteration and ove to the next loop iteration.

        #Defensuve range check
        if numeric_part <1 or numeric_part >395: #The constitution of India has 395 articles. If the numeric part is less than 1 or greater than 395, we skip it. This is a safety check to avoid extracting incorrect sections.
            continue

        #Extract article text
        start = matches[i].start() #Gets the starting position of the article header in the text.
        end = matches[i+1].start() if i+1 < len(matches) else len(full_text) #Gets the starting position of the next article header. If this is the last article, it uses the end of the text.

        articles_test = full_text[start:end].strip() #Extracts the text of the article by slicing the full text from the start to the end position. .strip() removes any leading or trailing whitespace.

        if article_no not in articles: #Checks if the article number is already in the dictionary. This is a safety check to avoid duplicates.
            articles[article_no] = articles_test #Store Article 
        #Validation Output
    print("Total Articles Extracted: ", len(articles)) #Prints the article number that was extracted.

    #Sorting Articles numerically because dictionary order might be random.
    sorted_articles = sorted(
            articles.keys(), #list of article identifiers,For example ->['1','10','100','11','12','2','3','31A','368']
            key = lambda x: int(re.match(r"\d+", x).group()) #key= something tells the sorted function how to custom sort the items. In this case, we are sorting based on the numeric part of the article number. re.match(r"\d+", x).group() extracts the numeric part of the article identifier, allowing us to sort them in proper numerical order (1, 2, 3, ..., 10, 11, ..., 31A, etc. lambda x: is an anonymous function allows us to write it inline. It takes x as input and returns the numeric part for sorting purposes.
        )

    print("First 10 Articles: ", sorted_articles[:10])
    print("Last 10 Articles: ", sorted_articles[-10:])

    # ------------------------------------------------------------
    # STEP 5 — CLAUSE-AWARE SPLITTING
    #This step converts Articles into Retrieval Chunks, which is what our vector database will later search.
    #We split the article into legal clauses. Each clause becomes a retrieval chunk.
    # ------------------------------------------------------------

    print("Splitting into clause-aware chunks...")

    clause_pattern = re.compile(
        r'(?m)(?:(?<=\n)|(?<=—))\s*\(\d+\)' #Numeric clauses, the regex portion: \(\d+\), (=opening bracket,\d+ means one or more digits, \) means closing bracket.For example:(1). (?<=\n) means the clause starts after a new line, Regex Feature called a lookbehind, only match if it appears after newline, prevents matching random parentheses in the text.. (?<=—) means the clause can also start after an dash. \s* allows for optional whitespace before the clause number
        r'|(?:(?<=\n))\s*\([a-z]\)' #Alphabetic clauses, the regex portion: \([a-z]\), (=opening bracket, [a-z] means a single lowercase letter, \) means closing bracket. For example:(a).
        r'|(?:(?<=\n))\s*\([ivx]+\)', #Roman numeral clauses, the regex portion: \([ivx]+\), (=opening bracket, [ivx]+ means one or more lowercase Roman numeral characters, \) means closing bracket. For example:(i), (ii), (iii), etc.
        re.IGNORECASE #This makes the regex case-insensitive, allowing it to match both uppercase and lowercase letters in the alphabetic clauses
    )
    #this will store all RAG chunks.
    chunks = []

    #Loop through each article and split into clauses
    #for article_no, article_text in articles.items(): # .item() returns key + value pairs from the articles dictionary. article_no will be the key (like "1", "2", "5A") and article_text will be the corresponding text.
    for article_no in sorted_articles:
        article_text = articles[article_no]    

        clause_matches = list(clause_pattern.finditer(article_text)) #.finditer() returns all mathches with postion and also includes their postion in the text.
    
    #if list is empty, it means no clauses were detected. In that case, we will treat the entire article as one chunk. This is a fallback mechanism to ensure that we don't lose any content even if the clause splitting fails for some articles.
    #f" : Insert variable inside string.
    #This is a standard RAG chunk format
    #This block ensures EVERY article produces at least one chunk since Articles without clauses would be LOST
        if not clause_matches:
            chunks.append({
                "chunk_id": f"Art_{article_no}", # Unique identifier for each chunk
                "parent_id": f"Art_{article_no}", #Which article this chunk belongs to
                "text": f"Article {article_no}: {article_text}",  #We are storing: the article number and the article text together in the chunk text. This is important because if we only store the article text without the number, we might lose important context. 
                "metadata": {                   #Extra structured information about the chunk. Later useful for: filtering, ranking and debugging
                    "article_no": article_no,
                    "clause_no": None
                }
            })
            continue #Skip rest of loop and move to next article. this is needed because Because after this block, code will try split into clauses But this article has no clauses.

        #Obj of this  clause: Take clause positions > Cut article into clause-level pieces>Create clean, contextual chunks
    
        seen_clause_ids = set() #Create an empty collection of unque values. This will help us track which clause identifiers we have already seen in the current article and thus sets automatically remove duplicates. 

        #Loop through each detected clause in the article. 
        #clause_matches contains all the positions of the clauses in the article text. 
        #Len() used because we need the index i, not just the value.
        for i in range(len(clause_matches)):
            #Identify which clause we are processing. 
            clause_id = clause_matches[i].group().strip() #clause_match[i] gets one match object, .group() extracts (1), (a) or (ii). .strip() removes any leading or trailing whitespace. so final result clause_id = "(1)"
        
            #checks if we have I already seen this clause
            if clause_id in seen_clause_ids:
                continue #Skip if duplicate
                     #(1) > not present > add > {(1)}
                     #(2) > not present > add > {(1),(2)}
                     #(1) > already present > skip

            #Store the clause_id so we remember we have already processed it
            seen_clause_ids.add(clause_id) #If new, Insert clause_id into the set

            #Define the boundaries of ONE clause
            start = clause_matches[i].start() # . start() returns position (index) in text
             #ternary condition. If there is a next clause, the end is the start of the next clause. If this is the last clause, the end is the end of the article text.
            #clause_matches[i+1].start() Position where next clause starts. else len(article_text): End of article
            end = clause_matches[i+1].start() if i+1 < len(clause_matches) else len(article_text)
        


            #Extract the exact clause text
            #Add structured context (Article + Clause)
            #[start:end]> String slicing
            clause_text = article_text[start:end].strip() 

            enriched_text = f"Article {article_no}. Clause {clause_id}: {clause_text}"

             #Store each clause as a structured unit (chunk) for retrieval + LLM
            chunks.append({
                "chunk_id": f"Art_{article_no}_{clause_id}",
                "parent_id": f"Art_{article_no}",
                "text": enriched_text,
                "metadata": {
                    "article_no": article_no,
                    "clause_no": clause_id
                }
            })
            

    print("Total chunks created: ", len(chunks))

    # ------------------------------------------------------------
    # STEP 6 — CRITICAL VALIDATION CHECKS
    # ------------------------------------------------------------

    critical_articles = ["32", "226", "368", "395"]

    for art in critical_articles:
        if art not in articles:
            print(f"WARNING: Article {art} missing!")

    #Check if ANY chunk contains the phrase "habeas corpus"
    habeas_found = any("habeas corpus" in c["text"].lower() for c in chunks)
    print("Contains 'habeas corpus'? ->", habeas_found)

    if not habeas_found:
        print("WARNING: Semantic anchor 'habeas corpus' not detected.")

    # ------------------------------------------------------------
    # STEP 7 — SAVE PRODUCTION JSON
    # ------------------------------------------------------------
    #Save the chunked data permanently into a file
    #output/constitution_chunks.json
    #"w" = write mode, create new file OR overwrite existing file.
    #encoding="utf-8" Supports special characters
    #with open(...) as f: Context manager that automatically handles file opening and closing. 

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False) #Convert Python object -> JSON -> write to file

    print("Saved JSON to:", JSON_PATH)

    print("="*60)
    print("SECTION 2 COMPLETE")
    print("="*60)

    return chunks
