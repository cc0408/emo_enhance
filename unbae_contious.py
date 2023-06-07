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
        3: "love",
        5: "surprise"
    }
    punc = string.punctuation

    dataset, num_labels = load_data(args)
    dataset = dataset['test']
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).cuda()
    model_checkpoint = f'/home/xuxi/emo_enhance/model/{args.model}_fineTuneModel.pth'
    print('Loading checkpoint: %s' % model_checkpoint)
    model.load_state_dict(torch.load(model_checkpoint))
    mlm_model = AutoModelForMaskedLM.from_pretrained(args.model, num_labels=num_labels).cuda()
    mlm_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model,do_lower_case=True)
    end_index = min(args.start_index + args.num_samples, len(dataset))
    use_model = SentenceTransformer("johngiorgi/declutr-small")
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        # N * D
    for idx in range(args.start_index, end_index):
        sentence = dataset[idx]['sentence']
        label = dataset[idx]['label']
        print(idx)
        print('label', int2label[label])
        #print(sentence)
        ori_ebd = use_model.encode([sentence])
        input = sentence.split()
        input_ids = tokenizer.encode(sentence)
        ma = model(torch.LongTensor(input_ids).cuda().unsqueeze(0)).logits[0][label]
        print('%.3f'%ma.item(), sentence)
        answer = sentence
        add_pos = -1
        for index in range(1,len(input_ids)-1):
            inserted_ids = input_ids.copy()
            inserted_ids.insert(index, 103)
            inserted_ids = torch.tensor([inserted_ids]).cuda()
            predictions = mlm_model(inserted_ids)
            v, predicted_index = predictions[0][0].topk(args.k, dim=-1)
            
            for mask_ids in predicted_index[index]:
                word = tokenizer.convert_ids_to_tokens(mask_ids.item())
                if word[:2]  == "##" or word in punc:
                    continue
                tmp = inserted_ids.clone()
                tmp[0][index] = mask_ids
                output_sentence = tokenizer.decode(tmp[0][1:-1].squeeze(0).cpu().tolist())
                opt_ebd = use_model.encode([output_sentence])
                similarity = 1 - cosine(ori_ebd, opt_ebd)
                if similarity > args.threshold:
                    pred = model(tmp).logits
                    #print(similarity, pred[0][label].item(), output_sentence)
                    if pred[0][label] > ma:
                        ma = pred[0][label]
                        answer = output_sentence
                        add_pos = index
        print('%.3f'%ma.item(), answer)
        #continue
        if add_pos == -1:
            continue
        raw =  tokenizer.encode(answer)
        forbidden_indices = [i for i in range(len(raw)) if i != add_pos]
        forbidden_indices = torch.tensor(forbidden_indices).cuda()
        inputs_embeds = embeddings[raw] # T * D
        inputs_embeds.requires_grad = True
        optimizer = torch.optim.Adam([inputs_embeds], lr=args.lr)
        for i in range(args.num_iters):
            optimizer.zero_grad()
            pred = model(inputs_embeds=torch.unsqueeze(inputs_embeds,0)).logits
            top_preds = pred.sort(descending=True)[1]
            correct = (top_preds[:, 0] == label).long()
            indices = top_preds.gather(1, correct.view(-1, 1))
            adv_loss = -(pred[:, label] - pred.gather(1, indices).squeeze() + args.kappa).clamp(min=0).mean()
            adv_loss.backward()
            word_emb = inputs_embeds.grad[add_pos]
            molen = (word_emb * word_emb).sum()
            inputs_embeds.grad /= molen
            inputs_embeds.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()
        dis = torch.pow(torch.unsqueeze(inputs_embeds,1)-torch.unsqueeze(embeddings,0),2).mean(dim=2)
        output_ids = dis.argmin(dim=1)
        print(output_ids)
        output_sentence = tokenizer.decode(output_ids)
        print(output_sentence)
        

        


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
    parser.add_argument("--lr", default=1e-2, type=float,
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