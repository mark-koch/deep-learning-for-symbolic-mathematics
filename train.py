import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data
from modules import Transformer
from data import SymbolicIntegrationDataset, collate_fn, get_validation_batches
from vocab import vocab, token2id, id2token
import argparse
import os
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', dest='data_paths', required=True, nargs='+', help='Paths of the dataset files')
parser.add_argument('-v', '--valid', dest='validation_paths', required=False, nargs='+',
                    help='Paths of the validation files')
parser.add_argument('-s', '--save', dest='save_dir', required=False, default='save', help='Where to save checkpoints')
parser.add_argument('-l', '--log', dest='log_dir', required=False, default='log', help='Where to save logs')
parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                    help='Checkpoint to continue training from')
parser.add_argument('--batch_size', dest='batch_size', required=False, default=32, type=int)
parser.add_argument('--print_iter', dest='print_iter', required=False, default=1, type=int,
                    help='Print progress every x iterations')
parser.add_argument('--plot_iter', dest='plot_iter', required=False, default=1000, type=int,
                    help='Plot attention every x iterations')
parser.add_argument('--plot_layers', dest='plot_layers', required=False, nargs="+",
                    default=["decoder_layer3_block2"],
                    help='List of layers to plot attention on')
parser.add_argument('--save_iter', dest='save_iter', required=False, default=10000, type=int,
                    help='Save checkpoint every x iterations')
parser.add_argument('--val_iter', dest='val_iter', required=False, default=5000, type=int,
                    help='Calculate validation accuracy every x iterations')
parser.add_argument('--use_amp', dest='use_amp', action='store_true', help='Use Apex Automatic Mixed Precision')

parser.add_argument('--num_encoder_layers', dest="num_encoder_layers", required=False, default=6, type=int)
parser.add_argument('--num_decoder_layers', dest="num_decoder_layers", required=False, default=6, type=int)
parser.add_argument('--d_model', dest="d_model", required=False, default=512, type=int)
parser.add_argument('--num_heads', dest="num_heads", required=False, default=8, type=int)
parser.add_argument('--d_ff', dest="d_ff", required=False, default=2048, type=int)
parser.add_argument('--dropout_rate', dest="dropout_rate", required=False, default=0.1, type=float)
parser.add_argument('--max_input_seq_len', dest="max_input_seq_len", required=False, default=128, type=int)
parser.add_argument('--max_target_seq_len', dest="max_target_seq_len", required=False, default=256, type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # Tensorboard
    writer = SummaryWriter(args.log_dir)

    print("Loading dataset...")
    dataset = SymbolicIntegrationDataset(args.data_paths, device=device, max_input_len=args.max_input_seq_len,
                                         max_target_len=args.max_target_seq_len)
    data_loader = torch.utils.data.DataLoader(dataset, args.batch_size, collate_fn=collate_fn)

    validation_data = {os.path.basename(path): get_validation_batches(path, args.batch_size,
                                                                      max_input_len=args.max_input_seq_len,
                                                                      max_target_len=args.max_target_seq_len)
                       for path in args.validation_paths}

    print("Loading transformer...")
    model = Transformer(num_encoder_layers=args.num_encoder_layers,
                        num_decoder_layers=args.num_decoder_layers,
                        d_model=args.d_model,
                        num_heads=args.num_heads,
                        d_ff=args.d_ff,
                        dropout_rate=args.dropout_rate,
                        max_input_seq_len=dataset.max_input_len,
                        max_target_seq_len=dataset.max_target_len,
                        input_vocab_size=len(vocab),
                        target_vocab_size=len(vocab),
                        vocab_padding_index=token2id("<pad>")).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    global_step = 0

    # Learning rate schedule
    # warmup_steps = 4000.0
    # def decay(_):
    #     step = global_step + 1
    #     return model.d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)

    if args.use_amp:
        import apex.amp as amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.restore_path is not None:
        print("Restoring from checkpoint: {}".format(args.restore_path))
        state = torch.load(args.restore_path, map_location=device)
        global_step = state["global_step"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        # scheduler.load_state_dict(state["scheduler"])
        print("Seeking previous position in dataset...")
        dataset.seek_forward(global_step * args.batch_size)

    def plot(m, p):
        fig, ax = plt.subplots()
        im = ax.imshow(m.detach().cpu())
        fig.colorbar(im)
        plt.title('{} Steps'.format(global_step))
        plt.savefig(p, format='png')
        plt.close(fig)

    print("Start training")
    while global_step < 100000000:
        for i, sample in enumerate(data_loader):
            optimizer.zero_grad()

            inp = sample["input"].to(device)                            # [batch_size, seq_len_enc]
            target = sample["target"].to(device)                        # [batch_size, seq_len_dec+1]
            enc_padding_mask = sample["enc_padding_mask"].to(device)    # [batch_size, 1, 1, seq_len_enc]
            dec_padding_mask = sample["dec_padding_mask"].to(device)    # [batch_size, 1, 1, seq_len_dec+1]
            dec_combined_mask = sample["dec_combined_mask"].to(device)  # [batch_size, 1, seq_len_dec+1, seq_len_dec+1]

            # Remove rightmost position target to feed to decoder
            target_to_decode = target[:, :-1]                      # [batch_size, seq_len_dec]
            dec_combined_mask = dec_combined_mask[:, :, :-1, :-1]  # [batch_size, 1, seq_len_dec, seq_len_dec]

            # Run Text2Mel. Shape of y_logits: [batch_size, seq_len_dec, target_vocab_size]
            y_logits, _, att = model(inp, target_to_decode, enc_padding_mask, dec_combined_mask)

            # Shift target one position to the left (remove first position). This is what the transformer should output
            target_for_loss = target[:, 1:]                   # [batch_size, seq_len_dec]
            dec_padding_mask = dec_padding_mask[:, 0, 0, 1:]  # This is needed for loss masking: [batch, dec_seq_len]

            # 'target_for_loss' contains integer indices of the words, while 'y_logits' has vectors with vocab_size many
            # components. Torch will automatically handle this, but we need to swap the 'target_vocab_size' axis with
            # the 'seq_len_dec' axis.
            # Resulting shape of loss is [batch_size, seq_len_dec]
            loss = nn.functional.cross_entropy(y_logits.transpose(1, 2),  # See comment above
                                               target_for_loss,
                                               reduction="none")  # We reduce manually later
            # We also should mask the loss before reducing
            loss *= (1 - dec_padding_mask.float())
            loss = loss.sum() / (dec_padding_mask.shape[0] * dec_padding_mask.shape[1] - dec_padding_mask.sum())

            if args.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            # scheduler.step()

            global_step += 1

            # Tensorboard
            writer.add_scalar('loss', loss, global_step)

            if global_step % args.print_iter == 0:
                print("Step {}, Loss={:.4f}".format(global_step, loss))

            # Plot attention
            if global_step % 1000 == 0:
                for layer in args.plot_layers:
                    # To label the axes we need differentiate between encoder, decoder and self attention
                    if "encoder" in layer:
                        q_labels, k_labels = inp, inp
                    elif "decoder" in layer and "block1" in layer:
                        q_labels, k_labels = target_for_loss, target_for_loss
                    else:
                        q_labels, k_labels = target_for_loss, inp
                    q_labels, k_labels = q_labels.detach().cpu(), k_labels.detach().cpu()  # [batch_size, seq_len]
                    q_labels = [id2token(q_labels[0, i].item()) for i in range(q_labels.shape[1])]
                    k_labels = [id2token(k_labels[0, i].item()) for i in range(k_labels.shape[1])]
                    # Plot each head
                    fig = plt.figure(figsize=(16, 8))
                    # plt.title("{} - Step {}".format(layer, global_step))
                    for head in range(args.num_heads):
                        attention = att[layer][0, head].detach().cpu().numpy()  # [seq_len_q, seq_len_k]
                        ax = fig.add_subplot(2, 4, head + 1)
                        ax.matshow(attention, cmap="viridis")
                        ax.set_xticks(range(len(k_labels)))
                        ax.set_yticks(range(len(q_labels)))
                        ax.set_ylim(len(k_labels) - 1.5, -0.5)
                        ax.set_xticklabels(k_labels, fontdict={"fontsize": 7}, rotation=90)
                        ax.set_yticklabels(q_labels, fontdict={"fontsize": 7})
                        ax.set_xlabel("Head {}".format(head + 1))
                    plt.tight_layout()
                writer.add_figure("attention_" + layer, fig, global_step)

            # Validation
            if global_step % args.val_iter == 0 and args.validation_paths:
                print("Calculating validation accuracy...")
                msg = "Validation accuracy:"
                model.eval()
                with torch.no_grad():
                    for name, samples in validation_data.items():
                        correct_total = 0
                        sequence_count = 0
                        for sample in samples:
                            # Run transformer on this batch. Exactly the same as above
                            inp = sample["input"].to(device)
                            target = sample["target"].to(device)
                            enc_padding_mask = sample["enc_padding_mask"].to(device)
                            dec_combined_mask = sample["dec_combined_mask"].to(device)
                            target_to_decode = target[:, :-1]
                            dec_combined_mask = dec_combined_mask[:, :, :-1, :-1]
                            _, y, _ = model(inp, target_to_decode, enc_padding_mask, dec_combined_mask)
                            # Apply padding mask to the output
                            mask = 1 - sample["dec_padding_mask"].to(device).float()  # [batch_size, 1, 1, seq_len_dec]
                            mask = mask[:, 0, 0, 1:].unsqueeze(-1)                    # [batch_size, seq_len_dec, 1]
                            y *= mask                                                 # [batch, seq_len_dec, vocab_size]
                            # Count the number of sequences that are correct
                            predictions = torch.argmax(y, dim=-1)             # [batch_size, seq_len_dec]
                            target = target[:, 1:]                            # [batch_size, seq_len_dec]
                            correct_positions = predictions == target         # [batch_size, seq_len_dec]
                            correct_sequences = correct_positions.sum(dim=1) == mask[:, :, 0].sum(dim=1)  # [batch_size]
                            num_correct = correct_sequences.sum()
                            correct_total += num_correct.item()
                            sequence_count += inp.shape[0]
                        accuracy = correct_total / sequence_count
                        print(correct_total, sequence_count, accuracy)
                        msg += " {}={:.2f}%,".format(name, accuracy * 100)
                        writer.add_scalar('validation_acc_' + name, accuracy, global_step)  # Tensorboard
                print(msg.rstrip(","))
                model.train()

            # Save checkpoints
            if global_step % args.save_iter == 0:
                state = {
                    "global_step": global_step,
                    "num_encoder_layers": model.num_encoder_layers,
                    "num_decoder_layers":  model.num_decoder_layers,
                    "d_model":  model.d_model,
                    "num_heads":  model.num_heads,
                    "d_ff":  model.d_ff,
                    "dropout_rate":  model.dropout_rate,
                    "vocab": vocab,
                    "vocab_padding_index": token2id("<pad>"),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # "scheduler": scheduler.state_dict(),
                }
                print("Saving checkpoint...")
                torch.save(state, os.path.join(args.save_dir, "checkpoint-{}.pth".format(global_step)))

