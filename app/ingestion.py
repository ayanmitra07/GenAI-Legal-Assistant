import os
import re
import subprocess
from app.config import REPO_URL, CLONE_DIR


# ============================================================
# SECTION 2 — PART TEXT INGESTION (PRODUCTION STABLE)
# ============================================================

def build_corpus():
    """
    Step 1: Clone / update repo
    Step 2: Load text files
    Step 3: Clean text
    Returns:
        full_text (str)
    """
    print("="*60)
    print("SECTION 2 — PART TEXT INGESTION (PRODUCTION STABLE)")
    print("="*60)

    # ------------------------------------------------------------
    # STEP 1 — CLONE OR UPDATE OFFICIAL TEXT REPOSITORY
    # ------------------------------------------------------------

    if not os.path.exists(CLONE_DIR): #Checks if the repo folder already exists
        print("Cloning Constitution repository...")
        subprocess.run(["git", "clone", REPO_URL, CLONE_DIR],check = True) #Python is basically typing this in my terminal automatically: git clone https://github.com/Constitution-of-India/tm.git.  Check = true: If the command fails → crash the program.
    else:
        print("Repository exists. Pulling latest changes...")
        subprocess.run(["git", "-C", CLONE_DIR, "pull"], check = True) #If the repo already exists, it will pull the latest changes. git -C constitution_tm pull: Go to the constitution_tm folder and update it.

    print("Repository is ready.")

    # ------------------------------------------------------------
    # STEP 2 — LOAD ALL PART TEXT FILES
    # ------------------------------------------------------------
    #os.listdir(CLONE_DIR):lists all files inside a directory. ->["PART_I.txt", "PART_II.txt", "README.md"]
    #if f.upper().startswith("PART"): We only want to read files that start with "PART" (ignoring README.md)
    #f.endswith(".txt"): We only want to read text files

    part_files = sorted([
        f for f in os.listdir(CLONE_DIR) #It is called a list comprehension. Loop through each file in the directory. -> f = "PART_I.txt", f = PART_II.txt,f = README.md
        if f.upper().startswith("PART") and f.endswith(".txt")
    ])

    if not part_files:
        raise Exception("No PART text files found in the repository.") #If no files are found, it will raise an error and stop the program. Because the AI system cannot work without data.

    print("Total PART files detected: ", len(part_files)) #length of list

    full_text = "" #This creates an empty string variable which will hold the entire constitution text.

    for file in part_files: #Loop through every text file.(file = "PART_I.txt", then file = "PART_II.txt", etc.)
        file_path = os.path.join(CLONE_DIR, file)

        with open(file_path, "r", encoding = "utf-8") as f: #os.path.join(CLONE_DIR, file) -> Creates a full path.for example: C:/project/constitution_tm/PART_I.txt. "r" means we are opening the file in read mode. encoding = "utf-8" ensures that we can read all characters properly. with open():Because Python automatically closes the file after reading and Safer for memory
            full_text += "\n" + f.read()#f.read():Reads the entire file content. "\n": adds a new line before the content of each file to ensure proper separation between parts. += appends the content of each file to the full_text variable.
    #Till this point we now we have the full constitution inside one variable.
    print("Total characters loaded: ", len(full_text)) #This counts the number of characters and tells us that the dataset loaded correctly.

    # ------------------------------------------------------------
    # STEP 3 — CLEAN TEXT (STRICT NORMALIZATION)
    # ------------------------------------------------------------

    #Remove form feed chracters, \x0c is a control character which is hidden and used to mark page breaks. .replace() is a string method that replaces all occurrences of a specified value with another value. 
    full_text = full_text.replace("\x0c", "")

    #Remove standalone page numbers. Regex is pattern matching. \n\d+\n means we are looking for a pattern where there is a new line, followed by one or more digits, and then another new line. re.sub() replaces this pattern with an empty string, effectively removing page numbers that are on their own lines.
    full_text = re.sub(r'\n\d+\n','\n',full_text) #re.sub(pattern, replacement, text) where r means raw string. \n means newline and \d meanss any digit

    #Remove header lines
    full_text = re.sub(r'\nTHE CONSTITUTION OF INDIA\n','\n', full_text) #This removes the header lines that say "THE CONSTITUTION OF INDIA" which appear multiple times in the text.

    #Normalize line breaks
    full_text = re.sub(r'\r\n', '\n', full_text) #This replaces any occurrence of \r\n (carriage return + newline). This ensures that line breaks are consistent throughout the text.
    full_text = re.sub(r'\n{3,}', '\n\n', full_text) #This fixes excess spacing. It looks for three or more consecutive newlines and replaces them with just two newlines.

    print("Text cleaning complete.")

    return full_text



