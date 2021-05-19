import argparse
import time

# from others.logging import init_logger
# from prepro import data_builder
from transformers import BertTokenizer

def process_chinese_ids(args):
    add_token_list = ['[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]']
    print(args.temp_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', cache_dir=args.temp_dir, do_lower_case=True, local_files_only=True, additional_special_tokens=add_token_list)
    gold_file = args.result_path + '.gold'
    gold_file_out = args.result_path + '.gold_chn'
    candi_file = args.result_path + '.candidate'
    candi_file_out = args.result_path + '.candidate_chn'
    print("gold_file = ", gold_file)
    print("candi_file = ", candi_file)
    print("gold_file_out = ", gold_file_out)
    print("candi_file_out = ", candi_file_out)
    cand_out = ""
    gold_out = ""

    with open(gold_file, 'r') as gold:
        with open(candi_file, 'r') as cand:
            cand_lines = cand.readlines()
            for i, each_gold in enumerate(gold.readlines()):
                print("step = ", i, end='\r')
                gold_ids = [int(each) for each in each_gold.split() if int(each) > 5 or int(each) == 3]
                gold_str = ' '.join(tokenizer.convert_ids_to_tokens(gold_ids)).replace(' ##', '')
                cand_ids = [int(each) for each in cand_lines[i].split() if int(each) > 5 or int(each) == 3]
                cand_str = ' '.join(tokenizer.convert_ids_to_tokens(cand_ids)).replace(' ##', '')
                # print("gold_str = ", gold_str)
                gold_out += (gold_str + '\n')
                cand_out += (cand_str + '\n')
                # print(gold_str)
                # print(cand_str)
                # exit()
    with open(gold_file_out, 'w') as writer:
        writer.write(gold_out)
    with open(candi_file_out, 'w') as writer:
        writer.write(cand_out)


    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../tmp')
    args = parser.parse_args()
    process_chinese_ids(args)

