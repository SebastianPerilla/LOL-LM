from datetime import datetime

def get_editor_prompt() -> str:
    """
    Returns the editor prompt template with current date awareness.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().strftime("%Y")
    
    return f"""You are a Satire News Editor. 
Today's date is {current_date} (Year: {current_year}).

Here are real news facts found online:
{{raw_news}}

Task: Identify ONE specific absurdity or contradiction in this news, keeping in mind that today is {current_date}.
Be aware of temporal inconsistencies - if the news mentions dates in the future relative to {current_date}, that's a red flag for satire.
Output a 'Satirical Premise' (1 sentence) and a 'Mock Headline'.
"""

