from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import transformers
import torch
import argparse
from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict

def main(args):
    num_labels = 6
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).cuda()
    model_checkpoint = f'/home/xuxi/emo_enhance/model/{args.model}_fineTuneModel.pth'
    print('Loading checkpoint: %s' % model_checkpoint)
    model.load_state_dict(torch.load(model_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(args.model,do_lower_case=True)
    input_ids = tokenizer.encode(args.sentence)
    clean_logit = model(input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda())[1].cpu()
    print(clean_logit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder for loading trained models")
    parser.add_argument("--gpt2_checkpoint_folder", default="result/", type=str,
        help="folder for loading GPT2 model trained with BERT tokenizer")
    parser.add_argument("--adv_samples_folder", default="adv_samples/", type=str,
        help="folder for saving generated samples")
    parser.add_argument("--dump_path", default="", type=str,
        help="Path to dump logs")

    # Data 
    parser.add_argument("--data_folder", default="data/", type=str,
        help="folder in which to store data")
    parser.add_argument("--dataset", default="emos", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2", "sst5", "emos"],
        help="classification dataset to use")
    parser.add_argument("--mnli_option", default="matched", type=str,
        choices=["matched", "mismatched"],
        help="use matched or mismatched test set for MNLI")
    parser.add_argument("--num_samples", default=1, type=int,
        help="number of samples to attack")
    parser.add_argument("--sentence", default="", type=str,
        help="sentence for eval")

    # Model
    parser.add_argument("--model", default="bert-base-uncased", type=str,
        help="type of model")
    parser.add_argument("--finetune", default=True, type=bool_flag,
        help="load finetuned model")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--num_iters", default=100, type=int,
        help="number of epochs to train for")
    parser.add_argument("--batch_size", default=10, type=int,
        help="batch size for gumbel-softmax samples")
    parser.add_argument("--attack_target", default="premise", type=str,
        choices=["premise", "hypothesis"],
        help="attack either the premise or hypothesis for MNLI")
    parser.add_argument("--initial_coeff", default=15, type=int,
        help="initial log coefficients")
    parser.add_argument("--adv_loss", default="cw", type=str,
        choices=["cw", "ce"],
        help="adversarial loss")
    parser.add_argument("--constraint", default="bertscore_idf", type=str,
        choices=["cosine", "bertscore", "bertscore_idf"],
        help="constraint function")
    parser.add_argument("--lr", default=3e-1, type=float,
        help="learning rate")
    parser.add_argument("--kappa", default=5, type=float,
        help="CW loss margin")
    parser.add_argument("--embed_layer", default=-1, type=int,
        help="which layer of LM to extract embeddings from")
    parser.add_argument("--lam_sim", default=1, type=float,
        help="embedding similarity regularizer")
    parser.add_argument("--lam_perp", default=1, type=float,
        help="(log) perplexity regularizer")
    parser.add_argument("--print_every", default=10, type=int,
        help="print loss every x iterations")
    parser.add_argument("--gumbel_samples", default=100, type=int,
        help="number of gumbel samples; if 0, use argmax")

    args = parser.parse_args()
    print_args(args)
    main(args)