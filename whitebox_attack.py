# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from bert_score.utils import get_idf_dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import transformers
import math
import argparse
import math
import jiwer
import numpy as np
import warnings
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time 
import torch
import torch.nn.functional as F
import csv
from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict
import json
from BERT_cls_lrp import BertForSequenceClassification as BertForSequenceClassificationTest
from ExplanationGenerator import Generator

def scores_per_word_from_scores_per_token(input, tokenizer, input_ids, scores_per_id):
    words = tokenizer.convert_ids_to_tokens(input_ids)
    words = [word.replace('##', '') for word in words]

    # TODO: DELETE
    input_ids_chars = []
    for word in words:
        if word in ['[CLS]', '[SEP]', '[UNK]', '[PAD]']:
            continue
        input_ids_chars += list(word)

    score_per_word = []
    start_idx = 0
    end_idx = 0
    # TODO: DELETE
    words_from_chars = []
    
    for inp in input:
        if start_idx >= len(scores_per_id):
            break
        lids = 0
        while len(inp) > lids:
            lids += len(words[end_idx])
            end_idx += 1
        score_per_word += [torch.max(scores_per_id[start_idx:end_idx]).item()] *(end_idx-start_idx)

        # TODO: DELETE
        words_from_chars.append(''.join(input_ids_chars[start_idx:end_idx]))

        start_idx = end_idx
    #print(scores_per_id,score_per_word)
    '''
    print(scores_per_id,score_per_word)
    if (words_from_chars[:-1] != input[:len(words_from_chars)-1]):
        print(words_from_chars)
        print(input[:len(words_from_chars)])
        print(words)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert False
    '''
    return torch.tensor(score_per_word)

def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def main(args):
    pretrained = args.model.startswith('textattack')
    output_file = get_output_file(args, args.model, args.start_index, args.start_index + args.num_samples)
    output_file = os.path.join(args.adv_samples_folder, output_file)
    print(f"Outputting files to {output_file}")
    if os.path.exists(output_file):
        print('Skipping batch as it has already been completed.')
        exit()
    wt = open('output.csv','w')
    writer = csv.writer(wt)
    # Load dataset
    dataset, num_labels = load_data(args)
    label_perm = lambda x: x
    if pretrained and args.model == 'textattack/bert-base-uncased-MNLI':
        label_perm = lambda x: (x + 1) % 3

    # Load tokenizer, model, and reference model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = 512
    test_classifier = BertForSequenceClassificationTest.from_pretrained("bert-base-uncased",num_labels=num_labels).cuda()
    model_save_file = os.path.join('bert_models/movies/classifier','classifier.pt')
    test_classifier.load_state_dict(torch.load(model_save_file))
    test_classifier.eval()
    explanations = Generator(test_classifier)
    model = explanations.generate_LRP
    models = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).cuda()
    if not pretrained:
        # Load model to attack
        suffix = '_finetune' if args.finetune else ''
        model_checkpoint = os.path.join(args.result_folder, '%s_%s%s.pth' % (args.model.replace('/', '-'), args.dataset, suffix))
        print('Loading checkpoint: %s' % model_checkpoint)
        models.load_state_dict(torch.load(model_checkpoint))
        tokenizer.model_max_length = 512
    if args.model == 'gpt2':
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
    if 'bert-base-uncase' in args.model:
        # for BERT, load GPT-2 trained on BERT tokenizer
        ref_model = load_gpt2_from_dict("%s/transformer_wikitext-103.pth" % args.gpt2_checkpoint_folder, output_hidden_states=True).cuda()
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).cuda()
    with torch.no_grad():
        embeddings = models.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        
    # encode dataset using tokenizer
    if args.dataset == "mnli":
        testset_key = "validation_%s" % args.mnli_option
        preprocess_function = lambda examples: tokenizer(
            examples['premise'], examples['hypothesis'], max_length=256, truncation=True)
    else:
        text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp", "sst2", "sst5"]) else 'sentence'
        testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp", "sst2", "sst5"]) else 'validation'
        preprocess_function = lambda examples: tokenizer(examples[text_key], max_length=256, truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
        
    # Compute idf dictionary for BERTScore
    if args.constraint == "bertscore_idf":
        if args.dataset == 'mnli':
            idf_dict = get_idf_dict(dataset['train']['premise'] + dataset['train']['hypothesis'], tokenizer, nthreads=20)
        else:
            idf_dict = get_idf_dict(dataset['train'][text_key], tokenizer, nthreads=20)
        
    if args.dataset == 'mnli':
        adv_log_coeffs = {'premise': [], 'hypothesis': []}
        clean_texts = {'premise': [], 'hypothesis': []}
        adv_texts = {'premise': [], 'hypothesis': []}
    else:
        adv_log_coeffs, clean_texts, adv_texts = [], [], []
    clean_logits = []
    adv_logits = []
    token_errors = []
    times = []
    succ = 0
    summ = 0
    #lbfile = csv.writer(open('lbfile.csv','w+'))
    assert args.start_index < len(encoded_dataset[testset_key]), 'Starting index %d is larger than dataset length %d' % (args.start_index, len(encoded_dataset[testset_key]))
    end_index = min(args.start_index + args.num_samples, len(encoded_dataset[testset_key]))
    adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters)
    #print(end_index)
    for idx in range(args.start_index, end_index):
        #if idx%100 == 0:
        print(idx)
        input_ids = encoded_dataset[testset_key]['input_ids'][idx]
        #lbfile.writerow([label_perm(encoded_dataset[testset_key]['label'][idx])])
        #continue
        print(tokenizer.convert_ids_to_tokens(input_ids))
        if args.model == 'gpt2':
            token_type_ids = None
        else:
            token_type_ids = encoded_dataset[testset_key]['token_type_ids'][idx]
        label = label_perm(encoded_dataset[testset_key]['label'][idx])
        clean_logit = models(input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
                             token_type_ids=(None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).cuda())).logits.data.cpu()
        print('LABEL')
        print(label)
        print('TEXT')
        print(tokenizer.decode(input_ids))
        print('LOGITS')
        print(clean_logit)
        
        offset = 0 if args.model == 'gpt2' else 1
        forbidden = np.ones(len(input_ids)).astype('bool')
        # set [CLS] and [SEP] tokens to forbidden
        important_fragment = []
        cam_target = model(input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(), index=label)[0]
        cam_target = cam_target.clamp(min=0)
        cam = cam_target
        inp = tokenizer.decode(input_ids).split() 
        #inp = [tokenizer.decode(i) for i in input_ids]
        #cam = scores_per_word_from_scores_per_token(inp, tokenizer,input_ids, cam)
        _, indices = cam.topk(k=max(1,min(5,len(input_ids)-2)))
        lowbd = _[-1]-1e-9
        cam = scores_per_word_from_scores_per_token(inp, tokenizer,input_ids, cam)
        for zz in range(len(cam)):
            if cam[zz] > lowbd:
                important_fragment.append(zz)
        #important_fragment=indices.tolist()
        #important_fragment = [i+1 for i in important_fragment]
        print(important_fragment)
        print(np.array(tokenizer.convert_ids_to_tokens(input_ids))[important_fragment])
        #important_fragment = sum([list(range(i[0],i[1])) for i in important_fragment],[])
        forbidden[important_fragment] = False
        forbidden[0] = True
        forbidden[-1] = True
        
        if args.dataset == 'mnli':
            # set either premise or hypothesis to forbidden
            premise_length = len(tokenizer.encode(encoded_dataset[testset_key]['premise'][idx]))
            input_ids_premise = input_ids[offset:(premise_length-offset)]
            input_ids_hypothesis = input_ids[premise_length:len(input_ids)-offset]
            if args.attack_target == "hypothesis":
                forbidden[:premise_length] = True
            else:
                forbidden[(premise_length-offset):] = True
        forbidden_indices = np.arange(0, len(input_ids))[forbidden]
        forbidden_indices = torch.from_numpy(forbidden_indices).cuda()
        token_type_ids_batch = (None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).repeat(args.batch_size, 1).cuda())

        start_time = time.time()
        with torch.no_grad():
            orig_output = ref_model(torch.LongTensor(input_ids).cuda().unsqueeze(0)).hidden_states[args.embed_layer]
            if args.constraint.startswith('bertscore'):
                if args.constraint == "bertscore_idf":
                    ref_weights = torch.FloatTensor([idf_dict[idx] for idx in input_ids]).cuda()
                    ref_weights /= ref_weights.sum()
                else:
                    ref_weights = None
            elif args.constraint == 'cosine':
                # GPT-2 reference model uses last token embedding instead of pooling
                if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                    orig_output = orig_output[:, -1]
                else:
                    orig_output = orig_output.mean(1)
            log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))
            indices = torch.arange(log_coeffs.size(0)).long()
            log_coeffs[indices, torch.LongTensor(input_ids)] = args.initial_coeff
            log_coeffs = log_coeffs.cuda()
            log_coeffs.requires_grad = True

        optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
        start = time.time()
        for i in range(args.num_iters):
            optimizer.zero_grad()
            coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1), hard=False) # B x T x V
            inputs_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D
            pred = models(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch).logits
            if args.adv_loss == 'ce':
                adv_loss = -F.cross_entropy(pred, label * torch.ones(args.batch_size).long().cuda())
            elif args.adv_loss == 'cw':
                top_preds = pred.sort(descending=True)[1]
                correct = (top_preds[:, 0] == label).long()
                indices = top_preds.gather(1, correct.view(-1, 1))
                adv_loss = (pred[:, label] - pred.gather(1, indices).squeeze() + args.kappa).clamp(min=0).mean()
            
            # Similarity constraint
            ref_embeds = (coeffs @ ref_embeddings[None, :, :])
            pred = ref_model(inputs_embeds=ref_embeds)
            if args.lam_sim > 0:
                output = pred.hidden_states[args.embed_layer]
                if args.constraint.startswith('bertscore'):
                    ref_loss = -args.lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
                else:
                    if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                        output = output[:, -1]
                    else:
                        output = output.mean(1)
                    cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
                    ref_loss = -args.lam_sim * cosine.mean()
            else:
                ref_loss = torch.Tensor([0]).cuda()
               
            # (log) perplexity constraint
            if args.lam_perp > 0:
                perp_loss = args.lam_perp * log_perplexity(pred.logits, coeffs)
            else:
                perp_loss = torch.Tensor([0]).cuda()
                
            # Compute loss and backward
            total_loss = -adv_loss + ref_loss + perp_loss
            total_loss.backward()
            
            entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
            if i % args.print_every == 0:
                print('Iteration %d: loss = %.4f, adv_loss = %.4f, ref_loss = %.4f, perp_loss = %.4f, entropy=%.4f, time=%.2f' % (
                    i+1, total_loss.item(), adv_loss.item(), ref_loss.item(), perp_loss.item(), entropy.item(), time.time() - start))

            # Gradient step
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()

            # Log statistics
            adv_losses[idx - args.start_index, i] = adv_loss.detach().item()
            ref_losses[idx - args.start_index, i] = ref_loss.detach().item()
            perp_losses[idx - args.start_index, i] = perp_loss.detach().item()
            entropies[idx - args.start_index, i] = entropy.detach().item()
        times.append(time.time() - start_time)
        
        print('CLEAN TEXT')
        if args.dataset == 'mnli':
            clean_premise = tokenizer.decode(input_ids_premise)
            clean_hypothesis = tokenizer.decode(input_ids_hypothesis)
            clean_texts['premise'].append(clean_premise)
            clean_texts['hypothesis'].append(clean_hypothesis)
            print('%s %s' % (clean_premise, clean_hypothesis))
        else:
            clean_text = tokenizer.decode(input_ids[offset:(len(input_ids)-offset)])
            clean_texts.append(clean_text)
            print(clean_text)
        clean_logits.append(clean_logit)
        
        print('ADVERSARIAL TEXT')
        ma = torch.full([1, num_labels],-1000000.1)
        unadv = ""
        choice_list = []
        with torch.no_grad():
            for j in range(args.gumbel_samples):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1).cpu().numpy()
                adv_ids = torch.from_numpy(adv_ids * (~forbidden) + input_ids * forbidden).cuda()
                if args.dataset == 'mnli':
                    if args.attack_target == 'premise':
                        adv_ids_premise = adv_ids[offset:(premise_length-offset)].cpu().tolist()
                        adv_ids_hypothesis = input_ids_hypothesis
                    else:
                        adv_ids_premise = input_ids_premise
                        adv_ids_hypothesis = adv_ids[premise_length:len(adv_ids)-offset].cpu().tolist()
                    adv_premise = tokenizer.decode(adv_ids_premise)
                    adv_hypothesis = tokenizer.decode(adv_ids_hypothesis)
                    x = tokenizer(adv_premise, adv_hypothesis, max_length=256, truncation=True, return_tensors='pt')
                    #token_errors.append(wer(input_ids_premise + input_ids_hypothesis, x['input_ids'][0]))
                else:
                    adv_ids = adv_ids[offset:len(adv_ids)-offset].cpu().tolist()
                    adv_text = tokenizer.decode(adv_ids)
                    x = tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')
                    #token_errors.append(wer(adv_ids, x['input_ids'][0]))
                adv_logit = models(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                                  token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None)).logits.data.cpu()
                #print(adv_logit.size(),type(label))
                #choice_list.append([adv_logit[0][label],adv_text])
                if adv_logit[0][label] > ma[0][label]:
                    ma = adv_logit
                    unadv = adv_text
                if j == args.gumbel_samples - 1:
                    if args.dataset == 'mnli':
                        #adv_texts['premise'].append(adv_premise)
                        #adv_texts['hypothesis'].append(adv_hypothesis)
                        print('%s %s' % (adv_premise, adv_hypothesis))
                    else:
                        adv_texts.append(unadv)
                        print(unadv)
                    #adv_logits.append(ma)
                    break
        #choice_list = list(set(choice_list))
        #choice_list.sort(key=lambda x: (-x[0]))
        #for i in range(3):
        #    print(choice_list[i][1])
        writer.writerow((clean_text,unadv,label))
        print("successfully attack %d, total %d"%(succ, summ))       
        # remove special tokens from adv_log_coeffs
        '''
        if args.dataset == 'mnli':
            adv_log_coeffs['premise'].append(log_coeffs[offset:(premise_length-offset), :].cpu())
            adv_log_coeffs['hypothesis'].append(log_coeffs[premise_length:(log_coeffs.size(0)-offset), :].cpu())
        else:
            adv_log_coeffs.append(log_coeffs[offset:(log_coeffs.size(0)-offset), :].cpu()) # size T x V
        '''
        print('')
        print('CLEAN LOGITS')
        print(clean_logit) # size 1 x C
        print('ADVERSARIAL LOGITS')
        print(ma)   # size 1 x C
            
    print("Token Error Rate: %.4f (over %d tokens)" % (sum(token_errors) / len(token_errors), len(token_errors)))
    '''
    torch.save({
        'adv_log_coeffs': adv_log_coeffs, 
        'adv_logits': torch.cat(adv_logits, 0), # size N x C
        'adv_losses': adv_losses,
        'adv_texts': adv_texts,
        'clean_logits': torch.cat(clean_logits, 0), 
        'clean_texts': clean_texts, 
        'entropies': entropies,
        'labels': list(map(label_perm, encoded_dataset[testset_key]['label'][args.start_index:end_index])), 
        'perp_losses': perp_losses,
        'ref_losses': ref_losses,
        'times': times,
        'token_error': token_errors,
    }, output_file)
    '''

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
    parser.add_argument("--data_folder", required=True, type=str,
        help="folder in which to store data")
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "sst2", "sst5"],
        help="classification dataset to use")
    parser.add_argument("--mnli_option", default="matched", type=str,
        choices=["matched", "mismatched"],
        help="use matched or mismatched test set for MNLI")
    parser.add_argument("--num_samples", default=1, type=int,
        help="number of samples to attack")

    # Model
    parser.add_argument("--model", default="gpt2", type=str,
        help="type of model")
    parser.add_argument("--finetune", default=False, type=bool_flag,
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
