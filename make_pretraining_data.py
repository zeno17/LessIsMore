# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:31:07 2021

"""
from cleaner_utils import super_cleaner
from preprocessing_utils import book_to_sentences, whole_word_MO_tokenization_and_masking
from preprocessing_utils import MODataset
from gutenberg.acquire import load_etext



def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--corpus-dir", required=True,
                      help="Location of pre-training text files.")
  parser.add_argument("--vocab-file", required=True,
                      help="Location of vocabulary file.")
  parser.add_argument("--output-dir", required=True,
                      help="Where to write out the tfrecords.")
  parser.add_argument("--max-seq-length", default=128, type=int,
                      help="Number of tokens per example.")
  parser.add_argument("--num-processes", default=1, type=int,
                      help="Parallelize across multiple processes.")
  parser.add_argument("--blanks-separate-docs", default=True, type=bool,
                      help="Whether blank lines indicate document boundaries.")

  # toggle lower-case
  parser.add_argument("--do-lower-case", dest='do_lower_case',
                      action='store_true', help="Lower case input text.")
  parser.add_argument("--no-lower-case", dest='do_lower_case',
                      action='store_false', help="Don't lower case input text.")

  # toggle strip-accents
  parser.add_argument("--do-strip-accents", dest='strip_accents',
                      action='store_true', help="Strip accents (default).")
  parser.add_argument("--no-strip-accents", dest='strip_accents',
                      action='store_false', help="Don't strip accents.")

  # set defaults for toggles
  parser.set_defaults(do_lower_case=True)
  parser.set_defaults(strip_accents=True)

  args = parser.parse_args()

  utils.rmkdir(args.output_dir)
  if args.num_processes == 1:
    write_examples(0, args)
  else:
    jobs = []
    for i in range(args.num_processes):
      job = multiprocessing.Process(target=write_examples, args=(i, args))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()


if __name__ == "__main__":
  main()