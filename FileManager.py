import os
import json
import zipfile
import shutil
import glob
import time  
import re    
from typing import Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
import pdfplumber
from transformers import pipeline
import PyPDF2
# from categorize_pdf import main as classify_pdfs_main

load_dotenv()
print("DEPLOYMENT_NAME:", os.getenv("DEPLOYMENT_NAME"))
print("OPENAI_API_BASE:", os.getenv("OPENAI_API_BASE"))
print("OPENAI_API_VERSION:", os.getenv("OPENAI_API_VERSION"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

class CopyFilesInput(BaseModel):
    extensions: str = Field(description="Comma-separated file extensions to copy (e.g., 'pdf,txt'). Use '*' for all files.", default="*")
    source_path: str = Field(description="The source directory to look for files.", default=".")
    destination: str = Field(description="The destination folder.")

class MoveFilesInput(BaseModel):
    file_type: str = Field(description="The file extension without dot (e.g., 'pdf', 'txt').")
    source_path: str = Field(description="The source directory to look for files.", default=".")
    destination: str = Field(description="The destination folder.")

class RenameItemInput(BaseModel):
    old_name: str = Field(description="The original filename or full path.")
    new_name: str = Field(description="The new filename or full path.")

class ListFilesInput(BaseModel):
    path: str = Field(description="The path to the directory to list files from.", default=".")

class ListFoldersInput(BaseModel):
    path: str = Field(description="The path to the directory to list folders from.", default=".")

class ScanFolderInput(BaseModel):
    path: str = Field(description="The path to the directory to scan.", default=".")

class CountFilesInput(BaseModel):
    file_type: str = Field(description="The file extension without dot (e.g., 'pdf', 'txt').")
    path: str = Field(description="The path to the directory to count files in.", default=".")

class ListAllItemsInput(BaseModel):
    path: str = Field(description="The path to the directory to list items from.", default=".")

class DirectoryInfoInput(BaseModel):
    path: str = Field(description="The path to the directory to get information about.", default=".")

class ZipFolderInput(BaseModel):
    folder_path: str = Field(description="The path to the folder to be zipped.")
    zip_path: str = Field(description="The output path for the zip file (including filename).", default=None)
    include_base_folder: bool = Field(description="Whether to include the base folder in the zip file.", default=True)

class CopyFilesRecursiveSafeInput(BaseModel):
    extensions: str = Field(
        description="Comma-separated file extensions to copy (e.g., 'jpg,png'). Use '*' for all files.",
        default="*"
    )
    source_path: str = Field(
        description="The source directory to look for files.",
        default="."
    )
    destination: str = Field(
        description="The destination folder to copy files to."
    )

class CategorizePDFsInput(BaseModel):
    source_path: str = Field(description="The source directory containing PDFs to categorize.")
    destination: str = Field(description="The destination directory to copy categorized PDFs.")


# Function to extract text from PDF
def extract_text(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to calculate score for a category
def calculate_score(text, keywords, patterns):
    text_lower = text.lower()
    keyword_count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
    
    pattern_count = 0
    for pattern in patterns:
        pattern_count += len(re.findall(pattern, text, re.IGNORECASE))
    
    if "•" in text or "-" in text or "*" in text:
        bullet_lines = sum(1 for line in text.split("\n") if re.match(r"^[•\-\*]\s", line.strip()))
        pattern_count += min(bullet_lines, 3)  # Cap at 3 to avoid overcounting
    
    return keyword_count + pattern_count

# Function to safely copy a file, renaming if necessary to avoid overwriting
def safe_copy(src, dst):
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    else:
        base, ext = os.path.splitext(dst)
        i = 1
        new_dst = f"{base}_{i}{ext}"
        while os.path.exists(new_dst):
            i += 1
            new_dst = f"{base}_{i}{ext}"
        shutil.copy(src, new_dst)


# Tools
@tool
def help():
    """Provides Users a guide, what operations they can perform """
    print("Available commands:")
    print("- List files in [path] (e.g., List files in C:\\Users\\Iqra R)")
    print("- List folders in [path]")
    print("- List all items in [path]")
    print("- Get directory info for [path]")
    print("- Scan folder [path]")
    print("- Move [file_type] files from [source] to [destination] (e.g., Move pdf files from . to ./PDFs)")
    print("- Count [file_type] files in [path]")
    print("- Delete file [file_path]")
    print("- Delete folder [folder_path]")
    print("- Rename [old_name] to [new_name]")
    print("- Check directories in [path]")
    print("- Change directory to [path]")
    print("- Get current directory")
    print("- Copy files from [source] to [destination] (optionally specify extensions, e.g., pdf,txt)")
    print("- Zip folder [folder_path] to [zip_path]")
    print("- Categorize PDFs in [source_path] to [destination_path]")
    print("Type 'exit' or 'quit' to close the program.")

@tool
def categorize_pdfs(input_data: CategorizePDFsInput) -> str:
    """Categorizes and copies PDFs from the source directory into categorized subfolders in the destination directory based on their content."""
    source_path = input_data.source_path
    destination = input_data.destination

    # # Initialize AzureChatOpenAI with environment variables
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,  # Deterministic output
        max_tokens=10  # Sufficient for category names
    )

    # Get list of PDF files in source_path
    pdf_files = [f for f in os.listdir(source_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        return "No PDF files found in the source path."

    # Define categories and create subfolders in destination
    categories = ['resume', 'invoice', 'medical report', 'research paper', 'novel', 'unknown']
    for category in categories:
        os.makedirs(os.path.join(destination, category), exist_ok=True)

    # Function to classify a single PDF
    def classify_pdf(file_path):
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() or ''  # Handle None from extract_text
            if not text.strip():
                return 'unknown'
            excerpt = text[:500]  # Limit to 500 characters for efficiency
            prompt = (
                f"Classify the following text into one of these categories: resume, invoice, "
                f"medical report, research paper, novel. Respond with only the category name.\n\n"
                f"{excerpt}\n\nClassification:"
            )
            response = llm.invoke(prompt)
            classification = response.content.strip().lower()
            return classification if classification in categories else 'unknown'
        except Exception:
            return 'unknown'  # Return 'unknown' on any error (e.g., encrypted PDF)

    # Function to safely copy files, handling name conflicts
    def safe_copy(src, dst):
        if not os.path.exists(dst):
            shutil.copy(src, dst)
        else:
            base, ext = os.path.splitext(dst)
            i = 1
            new_dst = f"{base}_{i}{ext}"
            while os.path.exists(new_dst):
                i += 1
                new_dst = f"{base}_{i}{ext}"
            shutil.copy(src, new_dst)

    # Process each PDF
    for pdf_file in pdf_files:
        file_path = os.path.join(source_path, pdf_file)
        classification = classify_pdf(file_path)
        target_folder = os.path.join(destination, classification)
        target_path = os.path.join(target_folder, pdf_file)
        safe_copy(file_path, target_path)

    return f"Categorized and copied {len(pdf_files)} PDFs to {destination}."

@tool
def copy_files_recursive(input_data: CopyFilesInput) -> str:
    """Copies files of specified extensions (or all files) from the source path and its subfolders to the destination folder."""
    extensions = input_data.extensions
    source_path = input_data.source_path
    destination = input_data.destination
    
    if extensions == '*':
        extensions = None  
    else:
        extensions = [ext.strip().lower() for ext in extensions.split(',')]
    
    os.makedirs(destination, exist_ok=True)
   
    copied_files = []
    
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            
            if extensions is None or any(file.lower().endswith('.' + ext) for ext in extensions):
                file_path = os.path.join(root, file)
                shutil.copy(file_path, os.path.join(destination, file))
                copied_files.append(file_path)
    
    
    if not copied_files:
        return f"No files found in {source_path} or its subfolders to copy."
    else:
        return f"Copied {len(copied_files)} files from {source_path} and its subfolders to {destination}."

@tool
def scan_folder(input_data: ScanFolderInput) -> str:
    """Scans the specified directory and returns file statistics."""
    path = input_data.path
    file_types = {}
    try:
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                ext = os.path.splitext(file)[-1].lower()
                ext = ext[1:] if ext else "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1
        return json.dumps(file_types)
    except Exception as e:
        return f"Error scanning folder {path}: {str(e)}"

@tool
def move_files(input_data: MoveFilesInput) -> str:
    """Moves files of a specific type from the source path to the destination folder."""
    file_type = input_data.file_type
    source_path = input_data.source_path
    destination = input_data.destination
    files = glob.glob(os.path.join(source_path, f"*.{file_type}"))
    if not files:
        return f"No {file_type} files found in {source_path} to move."
    os.makedirs(destination, exist_ok=True)
    for file in files:
        shutil.move(file, os.path.join(destination, os.path.basename(file)))
    return f"Moved {len(files)} {file_type} files from {source_path} to {destination}."

@tool
def count_files(input_data: CountFilesInput) -> str:
    """Counts files of a specific type in the specified folder."""
    file_type = input_data.file_type
    path = input_data.path
    files = glob.glob(os.path.join(path, f"*.{file_type}"))
    return f"There are {len(files)} {file_type} files in {path}."

@tool
def delete_file(file_path: str) -> str:
    """Deletes the specified file. Accepts full or relative paths."""
    if os.path.exists(file_path):
        os.remove(file_path)
        return f"Deleted {file_path}."
    else:
        return f"File {file_path} does not exist."

@tool
def delete_folder(folder_path: str) -> str:
    """Deletes the specified folder and its contents. Accepts full or relative paths."""
    if os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                return f"Deleted folder {folder_path} and its contents."
            except Exception as e:
                return f"Error deleting folder {folder_path}: {str(e)}"
        else:
            return f"{folder_path} is not a folder."
    else:
        return f"Folder {folder_path} does not exist."

@tool
def rename_item(input_data: RenameItemInput) -> str:
    """Renames a file or folder. Accepts full or relative paths."""
    try:
        old_name = input_data.old_name
        new_name = input_data.new_name
        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            return f"Renamed {old_name} to {new_name}."
        else:
            return f"{old_name} does not exist."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def list_files(input_data: ListFilesInput) -> str:
    """Lists ONLY files (not folders) in the specified directory."""
    path = input_data.path
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if files:
            return f"Files in {path}: {', '.join(files)}."
        else:
            return f"No files found in {path}."
    except Exception as e:
        return f"Error listing files in {path}: {str(e)}"

@tool
def list_folders(input_data: ListFoldersInput) -> str:
    """Lists ONLY folders/directories (not files) in the specified directory."""
    path = input_data.path
    try:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        if folders:
            return f"Folders in {path}: {', '.join(folders)}."
        else:
            return f"No folders found in {path}."
    except Exception as e:
        return f"Error listing folders in {path}: {str(e)}"

@tool
def list_all_items(input_data: ListAllItemsInput) -> str:
    """Lists both files AND folders in the specified directory."""
    path = input_data.path
    try:
        all_items = os.listdir(path)
        files = [f for f in all_items if os.path.isfile(os.path.join(path, f))]
        folders = [f for f in all_items if os.path.isdir(os.path.join(path, f))]
        result = ""
        if folders:
            result += f"Folders in {path}: {', '.join(folders)}.\n"
        else:
            result += f"No folders found in {path}.\n"
        if files:
            result += f"Files in {path}: {', '.join(files)}."
        else:
            result += f"No files found in {path}."
        return result
    except Exception as e:
        return f"Error listing items in {path}: {str(e)}"

@tool
def get_directory_info(input_data: DirectoryInfoInput) -> str:
    """Gets comprehensive information about the specified directory."""
    path = input_data.path
    try:
        all_items = os.listdir(path)
        files = [f for f in all_items if os.path.isfile(os.path.join(path, f))]
        folders = [f for f in all_items if os.path.isdir(os.path.join(path, f))]
        result = f"Directory Information for {path}:\n"
        result += f"Total items: {len(all_items)}\n"
        result += f"Number of folders: {len(folders)}\n"
        result += f"Number of files: {len(files)}\n\n"
        if folders:
            result += f"Folders: {', '.join(folders)}\n"
        else:
            result += "No folders found in the directory.\n"
        if files:
            result += f"Files: {', '.join(files)}"
        else:
            result += "No files found in the directory."
        return result
    except Exception as e:
        return f"Error getting directory info for {path}: {str(e)}"

@tool
def check_directories(input_data: ListFoldersInput) -> str:
    """Specifically checks for directories/folders in the specified location."""
    path = input_data.path
    try:
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        if folders:
            return f"Found {len(folders)} directories/folders in {path}: {', '.join(folders)}."
        else:
            return f"No directories/folders found in {path}."
    except Exception as e:
        return f"Error checking directories in {path}: {str(e)}"

@tool
def change_directory(path: str) -> str:
    """Changes the current working directory to the specified path."""
    try:
        os.chdir(path)
        return f"Changed current directory to {os.getcwd()}."
    except Exception as e:
        return f"Error changing directory to {path}: {str(e)}"

@tool
def get_current_directory() -> str:
    """Returns the current working directory."""
    return os.getcwd()

@tool
def zip_folder(input_data: ZipFolderInput) -> str:
    """Compresses a folder into a ZIP file with optional settings."""
    folder_path = input_data.folder_path
    include_base_folder = input_data.include_base_folder
    
    if not os.path.exists(folder_path):
        return f"Error: Folder '{folder_path}' does not exist."
    
    if not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a directory."
    
    if input_data.zip_path:
        zip_path = input_data.zip_path
    else:
        base_name = os.path.basename(folder_path)
        parent_dir = os.path.dirname(folder_path) or "."
        zip_path = os.path.join(parent_dir, f"{base_name}.zip")
    
    if not zip_path.lower().endswith('.zip'):
        zip_path += '.zip'
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            base_folder = os.path.basename(folder_path)
            
            total_files = 0
            for root, _, files in os.walk(folder_path):
                total_files += len(files)
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    if include_base_folder:
                        arcname = os.path.relpath(file_path, os.path.dirname(folder_path))
                    else:
                        arcname = os.path.relpath(file_path, folder_path)
                    
                    zipf.write(file_path, arcname)
        
        zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # Size in MB
        return f"Successfully created ZIP file '{zip_path}' ({zip_size:.2f} MB) containing {total_files} files from '{folder_path}'."
    
    except Exception as e:
        return f"Error creating ZIP file: {str(e)}"
    
# @tool
# def categorize_pdfs(input_data: CategorizePDFsInput) -> str:
#     """Categorizes and copies PDFs from the source directory into categorized subfolders in the destination directory based on their content."""
#     try:
#         num_pdfs = classify_pdfs_main(input_data.source_path, input_data.destination)
#         return f"Categorized {num_pdfs} PDFs from {input_data.source_path} to {input_data.destination}."
#     except Exception as e:
#         return f"Error categorizing PDFs: {str(e)}"
    
    
@tool
def copy_files_recursive_safe(input_data: CopyFilesRecursiveSafeInput) -> str:
    """Copies files of specified extensions (or all files) from the source path and its subfolders to the destination folder, preserving originals and handling name conflicts."""
    extensions = input_data.extensions
    source_path = input_data.source_path
    destination = input_data.destination
    
    if extensions == '*':
        extensions = None
    else:
        extensions = [ext.strip().lower() for ext in extensions.split(',')]
    
    os.makedirs(destination, exist_ok=True)
    
    copied_files = []
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if extensions is None or any(file.lower().endswith('.' + ext) for ext in extensions):
                file_path = os.path.join(root, file)
                target_path = os.path.join(destination, file)
                
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(os.path.join(destination, f"{name}_{i}{ext}")):
                        i += 1
                    target_path = os.path.join(destination, f"{name}_{i}{ext}")
                
                shutil.copy(file_path, target_path)
                copied_files.append(file_path)
    
    if not copied_files:
        return f"No files found in {source_path} or its subfolders to copy."
    else:
        return f"Copied {len(copied_files)} files from {source_path} and its subfolders to {destination}."

def invoke_with_retry(agent_chain, input_data, max_retries=2):
    """Invokes the agent chain with retry logic for 429 errors."""
    for attempt in range(max_retries):
        try:
            response = agent_chain.invoke(input_data)
            return response
        except Exception as e:
            error_str = str(e)
            if '429' in error_str:
                match = re.search(r'retry after (\d+) seconds?', error_str)
                wait_time = int(match.group(1)) if match else 1
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Max retries exceeded after encountering 429 errors.")

llm = AzureChatOpenAI(
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

def main():
    """Interactive agent-driven folder manager."""
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("DEPLOYMENT_NAME"),
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    agent_chain = initialize_agent(
        [
            list_folders,
            list_all_items,
            get_directory_info,
            list_files,
            scan_folder,
            move_files,
            count_files,
            delete_file,
            rename_item,
            delete_folder,
            check_directories,
            change_directory,
            get_current_directory,
            copy_files_recursive,
            zip_folder,
            copy_files_recursive_safe,
            categorize_pdfs,
            help
        ],
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=5
    )

    print("AI-POWERED FILE MANAGER!")
    print("NOTE: Actions like deleting files or folders are immediate and cannot be undone. Use with caution!")
    print("Type 'help' for available commands or 'exit' to quit.")

    while True:
        user_input = input("Enter command: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            response = invoke_with_retry(agent_chain, {"input": user_input})
            print(response["output"])
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()