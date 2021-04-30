# -*- coding: utf-8 -*-
"""

# Code originally from https://github.com/kiasar/gutenberg_cleaner but with changes from coreybobco and my own
# Given its so little code it was just easier to reuse it here in this way.
"""

from __future__ import absolute_import, unicode_literals
from builtins import str
import os
import re
import string
import numpy as np
from nltk import word_tokenize

TEXT_START_MARKERS = frozenset((
    "*END*THE SMALL PRINT",
    "*** START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
    "This etext was prepared by",
    "E-text prepared by",
    "Produced by",
    "Distributed Proofreading Team",
    "Proofreading Team at http://www.pgdp.net",
    "http://gallica.bnf.fr)",
    "      http://archive.org/details/",
    "http://www.pgdp.net",
    "by The Internet Archive)",
    "by The Internet Archive/Canadian Libraries",
    "by The Internet Archive/American Libraries",
    "public domain material from the Internet Archive",
    "Internet Archive)",
    "Internet Archive/Canadian Libraries",
    "Internet Archive/American Libraries",
    "material from the Google Print project",
    "*END THE SMALL PRINT",
    "***START OF THE PROJECT GUTENBERG",
    "This etext was produced by",
    "*** START OF THE COPYRIGHTED",
    "The Project Gutenberg",
    "http://gutenberg.spiegel.de/ erreichbar.",
    "Project Runeberg publishes",
    "Beginning of this Project Gutenberg",
    "Project Gutenberg Online Distributed",
    "Gutenberg Online Distributed",
    "the Project Gutenberg Online Distributed",
    "Project Gutenberg TEI",
    "This eBook was prepared by",
    "http://gutenberg2000.de erreichbar.",
    "This Etext was prepared by",
    "This Project Gutenberg Etext was prepared by",
    "Gutenberg Distributed Proofreaders",
    "Project Gutenberg Distributed Proofreaders",
    "the Project Gutenberg Online Distributed Proofreading Team",
    "**The Project Gutenberg",
    "*SMALL PRINT!",
    "More information about this book is at the top of this file.",
    "tells you about restrictions in how the file may be used.",
    "l'authorization à les utilizer pour preparer ce texte.",
    "of the etext through OCR.",
    "*****These eBooks Were Prepared By Thousands of Volunteers!*****",
    "We need your donations more than ever!",
    " *** START OF THIS PROJECT GUTENBERG",
    "****     SMALL PRINT!",
    '["Small Print" V.',
    '      (http://www.ibiblio.org/gutenberg/',
    'and the Project Gutenberg Online Distributed Proofreading Team',
    'Mary Meehan, and the Project Gutenberg Online Distributed Proofreading',
    '                this Project Gutenberg edition.',
    'Based on the Play by'
))

TEXT_END_MARKERS = frozenset((
    "*** END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "End of the Project Gutenberg",
    "End of The Project Gutenberg",
    "Ende dieses Project Gutenberg",
    "by Project Gutenberg",
    "End of Project Gutenberg",
    "End of this Project Gutenberg",
    "Ende dieses Projekt Gutenberg",
    "        ***END OF THE PROJECT GUTENBERG",
    "*** END OF THE COPYRIGHTED",
    "End of this is COPYRIGHTED",
    "Ende dieses Etextes ",
    "Ende dieses Project Gutenber",
    "Ende diese Project Gutenberg",
    "**This is a COPYRIGHTED Project Gutenberg Etext, Details Above**",
    "Fin de Project Gutenberg",
    "The Project Gutenberg Etext of ",
    "Ce document fut presente en lecture",
    "Ce document fut présenté en lecture",
    "More information about this book is at the top of this file.",
    "We need your donations more than ever!",
    "END OF PROJECT GUTENBERG",
    " End of the Project Gutenberg",
    " *** END OF THIS PROJECT GUTENBERG"
))

LEGALESE_START_MARKERS = frozenset(("<<THIS ELECTRONIC VERSION OF",))

LEGALESE_END_MARKERS = frozenset(("SERVICE THAT CHARGES FOR DOWNLOAD",))


EMPTY_PHRASES = frozenset(("a novel",
                           "by",
                           "to",
                           "\nby",
                           "and",
                           "for",
                           "for,",
                           "of",
                           "to",
                           "contents"))

END_MARKERS = frozenset(("the end",
                         "the end."))

TRANSCRIBER_NOTES = frozenset(("Minor typographical errors have been corrected without note. Dialect spellings have been retained.",
                               "Punctuation and the “long s” have been modernised; spelling has been retained as it appears in the original publication."
                               ))

GUTENBERG_DISCLAIMER = frozenset(("501(c)(3) educational corporation organized under the laws of the state of Mississippi and granted tax exempt status by the Internal Revenue Service. The Foundation's EIN or federal tax identification number is 64-6221541. Its 501(c)(3) letter is posted at http://pglaf.org/fundraising. Contributions to the Project Gutenberg Literary Archive Foundation are tax deductible to the full extent permitted by U.S. federal laws and your state's laws.",
                                 "The Foundation's principal office is located at 4557 Melan Dr. S. Fairbanks, AK, 99712., but its volunteers and employees are scattered throughout numerous locations. Its business office is located at 809 North 1500 West, Salt Lake City, UT 84116, (801) 596-1887, email business@pglaf.org. Email contact links and up to date contact information can be found at the Foundation's web site and official page at http://pglaf.org",
                                 'For additional contact information: Dr. Gregory B. Newby Chief Executive and Director gbnewby@pglaf.org',
                                 'Section 4. Information about Donations to the Project Gutenberg Literary Archive Foundation',
                                 'Project Gutenberg-tm depends upon and cannot survive without wide spread public support and donations to carry out its mission of increasing the number of public domain and licensed works that can be freely distributed in machine readable form accessible by the widest array of equipment including outdated equipment. Many small donations ($1 to $5,000) are particularly important to maintaining tax exempt status with the IRS.',
                                 'The Foundation is committed to complying with the laws regulating charities and charitable donations in all 50 states of the United States. Compliance requirements are not uniform and it takes a considerable effort, much paperwork and many fees to meet and keep up with these requirements. We do not solicit donations in locations where we have not received written confirmation of compliance. To SEND DONATIONS or determine the status of compliance for any particular state visit http://pglaf.org',
                                 'While we cannot and do not solicit contributions from states where we have not met the solicitation requirements, we know of no prohibition against accepting unsolicited donations from donors in such states who approach us with offers to donate.',
                                 'International donations are gratefully accepted, but we cannot make any statements concerning tax treatment of donations received from outside the United States. U.S. laws alone swamp our small staff.',
                                 'Please check the Project Gutenberg Web pages for current donation methods and addresses. Donations are accepted in a number of other ways including checks, online payments and credit card donations. To donate, please visit: http://pglaf.org/donate',
                                 'Section 5. General Information About Project Gutenberg-tm electronic works.',
                                 'Professor Michael S. Hart is the originator of the Project Gutenberg-tm concept of a library of electronic works that could be freely shared with anyone. For thirty years, he produced and distributed Project Gutenberg-tm eBooks with only a loose network of volunteer support.',
                                 'Project Gutenberg-tm eBooks are often created from several printed editions, all of which are confirmed as Public Domain in the U.S. unless a copyright notice is included. Thus, we do not necessarily keep eBooks in compliance with any particular paper edition.',
                                 "Each eBook is in a subdirectory of the same number as the eBook's eBook number, often in several formats including plain vanilla ASCII, compressed (zipped), HTML and others.",
                                 'Corrected EDITIONS of our eBooks replace the old file and take over the old filename and etext number. The replaced older file is renamed. VERSIONS based on separate sources are treated as new eBooks receiving new filenames and etext numbers.',
                                 'Most people start at our Web site which has the main PG search facility:',
                                 'http://www.gutenberg.org',
                                 'This Web site includes information about Project Gutenberg-tm, including how to make donations to the Project Gutenberg Literary Archive Foundation, how to help produce our new eBooks, and how to subscribe to our email newsletter to hear about new eBooks.',
                                 'EBooks posted prior to November 2003, with eBook numbers BELOW #10000, are filed in directories based on their release date. If you want to download any of these eBooks directly, rather than using the regular search system you may utilize the following addresses and just download by the etext year.',
                                 'http://www.ibiblio.org/gutenberg/etext06',
                                 '(Or /etext 05, 04, 03, 02, 01, 00, 99, 98, 97, 96, 95, 94, 93, 92, 92, 91 or 90)',
                                 'EBooks posted since November 2003, with etext numbers OVER #10000, are filed in a different way. The year of a release date is no longer part of the directory path. The path is based on the etext number (which is identical to the filename). The path to the file is made up of single digits corresponding to all but the last digit in the filename. For example an eBook of filename 10234 would be found at:',
                                 'http://www.gutenberg.org/1/0/2/3/10234',
                                 'or filename 24689 would be found at: http://www.gutenberg.org/2/4/6/8/24689',
                                 'An alternative method of locating eBooks: http://www.gutenberg.org/GUTINDEX.ALL',
                                 '*** END: FULL LICENSE ***',
                                 "501(c)(3) educational corporation organized under the laws of the state of Mississippi and granted tax exempt status by the Internal Revenue Service. The Foundation's EIN or federal tax identification number is 64-6221541. Its 501(c)(3) letter is posted at https://pglaf.org/fundraising. Contributions to the Project Gutenberg Literary Archive Foundation are tax deductible to the full extent permitted by U.S. federal laws and your state's laws.",
                                 'The Foundation is committed to complying with the laws regulating charities and charitable donations in all 50 states of the United States. Compliance requirements are not uniform and it takes a considerable effort, much paperwork and many fees to meet and keep up with these requirements. We do not solicit donations in locations where we have not received written confirmation of compliance. To SEND DONATIONS or determine the status of compliance for any particular state visit https://pglaf.org',
                                 'Please check the Project Gutenberg Web pages for current donation methods and addresses. Donations are accepted in a number of other ways including including checks, online payments and credit card donations. To donate, please visit: https://pglaf.org/donate',
                                 'Professor Michael S. Hart was the originator of the Project Gutenberg-tm concept of a library of electronic works that could be freely shared with anyone. For thirty years, he produced and distributed Project Gutenberg-tm eBooks with only a loose network of volunteer support.',
                                 'https://www.gutenberg.org',
                                 'a team of about twenty Project Gutenberg volunteers.',
                                 "501(c)(3) educational corporation organized under the laws of the state of Mississippi and granted tax exempt status by the Internal Revenue Service. The Foundation's EIN or federal tax identification number is 64-6221541. Contributions to the Project Gutenberg Literary Archive Foundation are tax deductible to the full extent permitted by U.S. federal laws and your state's laws.",
                                 "The Foundation's business office is located at 809 North 1500 West, Salt Lake City, UT 84116, (801) 596-1887. Email contact links and up to date contact information can be found at the Foundation's website and official page at www.gutenberg.org/contact",
                                 'Project Gutenberg-tm depends upon and cannot survive without widespread public support and donations to carry out its mission of increasing the number of public domain and licensed works that can be freely distributed in machine-readable form accessible by the widest array of equipment including outdated equipment. Many small donations ($1 to $5,000) are particularly important to maintaining tax exempt status with the IRS.',
                                 'The Foundation is committed to complying with the laws regulating charities and charitable donations in all 50 states of the United States. Compliance requirements are not uniform and it takes a considerable effort, much paperwork and many fees to meet and keep up with these requirements. We do not solicit donations in locations where we have not received written confirmation of compliance. To SEND DONATIONS or determine the status of compliance for any particular state visit www.gutenberg.org/donate',
                                 'Please check the Project Gutenberg web pages for current donation methods and addresses. Donations are accepted in a number of other ways including checks, online payments and credit card donations. To donate, please visit: www.gutenberg.org/donate',
                                 'Professor Michael S. Hart was the originator of the Project Gutenberg-tm concept of a library of electronic works that could be freely shared with anyone. For forty years, he produced and distributed Project Gutenberg-tm eBooks with only a loose network of volunteer support.',
                                 'Most people start at our website which has the main PG search facility: www.gutenberg.org',
                                 'This website includes information about Project Gutenberg-tm, including how to make donations to the Project Gutenberg Literary Archive Foundation, how to help produce our new eBooks, and how to subscribe to our email newsletter to hear about new eBooks.'))


def super_cleaner(book: str, min_token: int = 5, max_token: int = 600, mark_deletions: bool = False, verify_deletions=False, return_list=True) -> str:
    """
    Super clean the book (titles, footnotes, images, book information, etc.). may delete some good lines too.
    ^_^ Do you have a comment to make it better? make an issue here: https://github.com/kiasar/gutenberg_cleaner ^_^.
    IMPORTANT: if you don't want the text to be tokenize, just put min_token = -1.
    :rtype: str
    :param book: str of a gutenberg's book.
    :param min_token: The minimum tokens of a paragraph that is not "dialog" or "quote",
     -1 means don't tokenize the txt (so it will be faster).
    :param max_token: The maximum tokens of a paragraph.
    :return: str of the book with paragraphs that have been deleted are shown with "[deleted]" in it.
    you can split the book to paragraphs by "\n\n".
    """
    headless_book = _strip_headers(book)
    if '\n\n' in headless_book: # paragraphs are split with \n\n
        paragraphs = headless_book.replace('\r',"").split("\n\n")  # split the book to paragraphs.
    else: #paragraphs are split with \r\n\r\n
        paragraphs = [re.sub(' +', ' ', x.replace('\r\n', " ")) for x in headless_book.split("\r\n\r\n")]  # split the book to paragraphs.

    paragraphs_after_cleaning = []
    after_the_end = False
    after_the_index = False
    for par in paragraphs:
        if after_the_end or after_the_index:
            if verify_deletions:
                print(True, par)
                continue
            else:
                break
        if par.strip('\n').lower() in END_MARKERS:
            after_the_end = True
        if par.strip('\n').lower() == 'index' :
            after_the_index = True
            
        if verify_deletions:
            manual_verify_deletions(par)
        if _is_image(par) or _is_footnote(par) or _is_email_init(par) or \
                _is_books_copy(par) or _is_table(par) or _is_title_or_etc(par, min_token, max_token) or \
                    _is_table_of_contents(par) or _is_illustration(par) or _is_transcriber_notes(par):
                        
            if mark_deletions:
                paragraphs_after_cleaning.append("[deleted]")  # if the paragraph is not good, replace it with [deleted]
        else:
            #replace some final unnecessary stuff
            cleaned_text = strip_makeup(par)
            paragraphs_after_cleaning.append(cleaned_text)
    if return_list:    
        return list(np.unique(paragraphs_after_cleaning)) # joining the list of paragraphs into one string
    else:
        return " ".join(paragraphs_after_cleaning)


def strip_makeup(par):
    cleaned_text = par.replace('“', '"').replace('”', '"').replace("\n", " ")
    for match in re.findall(r'(\+.*?\+)', cleaned_text, flags=re.IGNORECASE):
        cleaned_text = cleaned_text.replace(match, match.strip('+'))
    for match in re.findall(r'(\_.*?\_)', cleaned_text, flags=re.IGNORECASE):
        cleaned_text = cleaned_text.replace(match, match.strip("_"))
    return cleaned_text
    
    
def manual_verify_deletions(par):
    print(_is_image(par) or _is_footnote(par) or _is_email_init(par) or _is_books_copy(par) \
          or _is_table(par) or _is_title_or_etc(par, -1, 600) or _is_table_of_contents(par) or \
              _is_illustration(par) or _is_transcriber_notes(par),
          par)
        
        
def _strip_headers(text):
    """Remove lines that are part of the Project Gutenberg header or footer.
    Note: The original version of the code can be found at:
    https://github.com/c-w/gutenberg/blob/master/gutenberg/cleanup/strip_headers.py
    Args:
        text (unicode): The body of the text to clean up.
    Returns:
        unicode: The text with any non-text content removed.
    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out)

email_regex = re.compile("[\w.-]+@[\w.-]+\.\w+")  # Regex to find Emails.
footnote_notation_regex = re.compile("^\{.+\}|^\[.+\]")  # Regex to find start of footnotes.
number_of_copies_regex = re.compile("[0-9]* copies|copyright")  # Regex to find copy mentioning.
starts_with_regex = re.compile('^[%_<>*]')  # If the text is started with these, it is not a good one.
image_formats_regex = re.compile("\.png|\.jpg|\.jpeg|\.gif|picture:")  # Regex to find images.


def _is_title_or_etc(text: str, min_token: int = -1, max_token: int = 600) -> bool:
    """
    determining if a paragraph is title or information of the book.
    IMPORTANT: if you don't want the text to be tokenize, just put min_token = -1.
    :rtype: bool
    :param text: Raw paragraph.
    :param min_token: The minimum tokens of a paragraph that is not "dialog" or "quote",
     -1 means don't tokenize the txt (so it will be faster).
    :param max_token: The maximum tokens of a paragraph.
    :return: Boolean, True if it is title or information of the book or a bad paragraph.
    """
    txt = text.strip()
    num_token = len(word_tokenize(txt)) if min_token >= 0 else -1
    if num_token > max_token:
        return True
    if len(txt) == 0 or num_token < min_token and not (txt.count('"') == 2 or txt.count('\'') == 2 or txt[-1] == ":"):
        return True  # Length is short but not "dialog" or "quote"
    if sum(1 for c in txt if c.isupper() or c.isdigit() or c in string.punctuation.replace("\"", "")) \
            / len(txt.replace(" ", "")) > 0.6:
        return True  # More than 60% of chars are UPPER or digits or punctuations so it might be title or etc.
    if txt.lower().startswith("appendix") or bool(re.search(starts_with_regex, txt)):
        return True
    if txt.count(":") > 3 and 2 * txt.count(":") - txt.count("\"") > 3:
        return True  # mostly information about the book.
    if ("@" in txt and len(txt) < 100) or ('printed in' in txt.lower() and len(txt) < 200) or "inc." in txt.lower() \
            or ('original title' in txt.lower() and len(txt) < 200):
        return True
    if text.strip().lower() in EMPTY_PHRASES:
        return True
    if sum([x[0].strip('\n').isupper() for x in text.split(' ') if len(x) > 0 ])/len([x for x in text.split(' ') if x != '']) > 0.6: #more than 75% of the words start with a capital letter.
        return True
    
    return False


def _is_table(text: str) -> bool:
    """
    determining if a paragraph is a table or catalog.
    :rtype: bool
    :param text: Raw paragraph.
    :return:  Boolean, True if it is a table or catalog.
    """
    txt = text.strip()
    if txt.count("   ") > 3 or txt.count("\t") > 2:
        txt = " ".join([line.strip() for line in txt.split("\n")])
        if txt.count("   ") > 3 or txt.count("\t") > 2:
            return True  # mostly tables.
    if txt.count("*") > 3 or txt.count("=") > 2:
        return True  # mostly catalogs and etc.
    
    return False


def _is_image(text: str) -> bool:
    """
    determining if a paragraph is for mentioning an image.
    :param text: Raw paragraph.
    :return: Boolean, True if it is for mentioning an image.
    """
    return bool(re.search(image_formats_regex, text.lower()))


def _is_footnote(text: str) -> bool:
    """
    determining if a paragraph is the footnote of the book.
    :rtype: bool
    :param text: Raw paragraph.
    :return: Boolean, True if it is the footnote of the book.
    """
    txt = text.strip()
    if "footnote" in txt.lower() and len(txt.replace(" ", "")) < 50:
        return True
    if "Transcriber’s Note:" in txt:
        return True
    if txt.strip() in GUTENBERG_DISCLAIMER:
        return True
    return bool(re.search(footnote_notation_regex, txt))  # if a line starts with {...} it might be a footnote.


def _is_books_copy(text: str) -> bool:
    """x
    determining if a paragraph indicates the number of copies of this book.
    :rtype: bool
    :param text: text: Raw paragraph.
    :return: Boolean, True if it is indicating the copy of book or copyrights.
    """
    if bool(re.search(number_of_copies_regex, text)) and len(text.replace(" ", "")) < 500:
        return True
    return False


def _is_email_init(text: str) -> bool:
    """
    determining if a paragraph includes an Email.
    :rtype: bool
    :param text: Raw paragraph.
    :return: Boolean, True if it includes an Email.
    """
    return bool(re.search(email_regex, text))

def _is_table_of_contents(text: str) -> bool:
    """
    Other functions were sometimes missing specific lines from the table of contents 
    
    check if sentence:
        contains 'CHAPTER'
        contains roman numerals (often used in )
    """
    
    if 'CHAPTER' in text:
        return True
    if "Part" in text and len(text.split(' ')) < 4:
        return True
    if _is_roman_numerals(text.split('.')[0].strip()) or _is_roman_numerals(text.split('.')[0].strip().strip('.')):
        return True
    return False

def _is_illustration(text: str) -> bool:
    return text.startswith('[Illustration:')

        
def _is_roman_numerals(text: str) -> bool:
    for char in text:
        if char not in ["M", "D", "C", "L", "X", "V", "I"]:
            return False
    return True

def _is_transcriber_notes(text: str) -> bool:
    if text in TRANSCRIBER_NOTES:
        return True
    return False