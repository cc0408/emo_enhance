from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
import transformers
import torch
import argparse
import string
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict

def main(args):
    int2label = {
        4: "sadness",
        2: "joy",
        0: "anger",
        1: "fear",
        5: "surprise"
    }
    punc = string.punctuation

    dataset, num_labels = load_data(args)
    dataset = dataset['test']
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).cuda()
    model_checkpoint = '/home/xuxi/emo_enhance/model/fineTuneModel.pt'
    print('Loading checkpoint: %s' % model_checkpoint)
    model.load_state_dict(torch.load(model_checkpoint))
    mlm_model = AutoModelForMaskedLM.from_pretrained(args.model, num_labels=num_labels).cuda()
    mlm_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model,do_lower_case=True)
    end_index = min(args.start_index + args.num_samples, len(dataset))
    index = 7
    use_model = SentenceTransformer("johngiorgi/declutr-small")
    
    for idx in range(args.start_index, end_index):
        sentence = dataset[idx]['sentence']
        label = dataset[idx]['label']
        print(idx)
        print('label', int2label[label])
        #print(sentence)
        ori_ebd = use_model.encode([sentence])
        input = sentence.split()
        input_ids = tokenizer.encode(sentence)
        clean_logits = model(torch.LongTensor(input_ids).cuda().unsqueeze(0)).logits[0][label].item()
        print(clean_logits, sentence, sep='\n')
        answer = sentence
        add_pos = []
        for index in range(1,len(input_ids)-1):
            inserted_ids = input_ids.copy()
            inserted_ids.insert(index, 103)
            inserted_ids = torch.tensor([inserted_ids]).cuda()
            predictions = mlm_model(inserted_ids)
            v, predicted_index = predictions[0][0].topk(args.k, dim=-1)
            
            lidx=len(input_ids)
            choice = []
            for mask_ids in predicted_index[index]:
                word = tokenizer.convert_ids_to_tokens(mask_ids.item())
                if word[:2]  == "##" or word in punc:
                    continue
                tmp = inserted_ids.clone()
                tmp[0][index] = mask_ids
                output_sentence = tokenizer.decode(tmp[0][1:-1].squeeze(0).cpu().tolist())
                opt_ebd = use_model.encode([output_sentence])
                similarity = 1 - cosine(ori_ebd, opt_ebd)
                if similarity > args.threshold - (lidx<=10)*0.3:
                    pred = model(tmp).logits
                    choice.append((pred[0][label].item(), mask_ids))
            choice.sort(reverse=True)
            if len(choice) != 0:
                add_pos.append(choice[0]+(index,))
        add_pos.sort(reverse=True)
        add_pos = add_pos[:min(len(add_pos),2)]
        add_pos.sort(key=lambda x:(x[1]),reverse=True)
        inserted_ids = input_ids.copy()
        for tp in add_pos:
            inserted_ids.insert(tp[2], tp[1])
        answer = tokenizer.decode(inserted_ids)
        unadv_logits = model(torch.LongTensor(inserted_ids).cuda().unsqueeze(0)).logits[0][label].item()
        print(unadv_logits, answer,sep='\n')
        

        


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
    parser.add_argument("--k", default=20, type=int,
        help="topk words")
    parser.add_argument("--threshold", default=0.99, type=float,
        help="threshold of use")
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