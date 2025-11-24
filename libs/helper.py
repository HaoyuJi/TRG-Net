import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from libs.class_id_map import get_id2class_map
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor
from tqdm import tqdm

from libs.tools import segment_video_labels, gen_label, generate_segment_features, create_logits


def train(
    train_loader: DataLoader,
    model: nn.Module,
    action_embeddings_data,
    action_embeddings_graph,
    joint_embeddings_graph,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    criterion_contrast: nn.Module,
    criterion_mse: nn.Module,
    lambda_bound_loss: float,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e") #init loss

    # switch training mode
    model.train()

    for sample in tqdm(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]

        x = x.to(device)
        t = t.to(device)
        b = b.to(device)
        mask = mask.to(device)

        batch_size = x.shape[0]

        # compute output and loss
        output_cls, output_bound, output_feature, logit_scale = model(x, mask, joint_embeddings_graph)

        # Absolute contrastive loss
        t_segment = segment_video_labels(t)

        label = [i[0] for seg in t_segment for i in seg]

        label_g = gen_label(label)  # （N，N）contrastive GT

        action_embedding = list()
        for single_label in label:
            action_item = action_embeddings_data[single_label].unsqueeze(dim=0)  # get embeddings
            action_embedding.append(action_item)

        action_embedding = torch.cat(action_embedding).cuda(device)

        action_features = []
        if isinstance(output_feature, list):
            # for i in range(len(output_feature)):
            action_feature = generate_segment_features(output_feature[0], t_segment, device)
            action_features.append(action_feature)

        # Relative Distance loss
        action_mask = ~torch.eye(len(label), device=device, dtype=torch.bool)

        action_embedding_graph = torch.zeros((len(label), len(label)), device=device)
        for i in range(len(label)):
            for j in range(len(label)):
                action_embedding_graph[i, j] = action_embeddings_graph[label[i], label[j]]

        action_feature = generate_segment_features(output_feature[0], t_segment, device)

        action_features_differences = action_feature.unsqueeze(1) - action_feature.unsqueeze(0)

        action_features_distance = torch.norm(action_features_differences, p=2, dim=-1)

        # print(action_feature)
        action_features_distance_normalized = (action_features_distance - action_features_distance.min()) / (
                action_features_distance.max() - action_features_distance.min())

        # action_features_graph = 1 - action_features_distance_normalized[action_mask]
        action_features_graph = 1 - action_features_distance_normalized
 
        loss = 0.0
        if isinstance(output_cls, list):
            n = len(output_cls)
            for out in output_cls:
                loss += criterion_cls(out, t, x) / n #action segmentation loss
        else:
            loss += criterion_cls(output_cls, t, x)

        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n #boundary loss
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        if isinstance(action_features, list):
            # for i in range(len(action_features)):
            logits_per_image, logits_per_text = create_logits(action_features[0], action_embedding, logit_scale)  # 2 sim matrix
            ground_truth = torch.tensor(label_g, dtype=action_features[0].dtype, device=device)  # GT

            loss_imgs = criterion_contrast(logits_per_image, ground_truth)
            loss_texts = criterion_contrast(logits_per_text, ground_truth)

            loss += 0.8 * ((loss_imgs + loss_texts) / 2)  # absolute contrastive loss

        language_loss = criterion_contrast(action_features_graph, action_embedding_graph)
        loss += 1.0 * language_loss

        # record loss
        losses.update(loss.item(), batch_size) #record loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    joint_embeddings_graph,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    device: str,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    boundary_th: float,
    tolerance: int,
    refinement_method: Optional[str] = None
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    losses = AverageMeter("Loss", ":.4e")
    postprocessor = PostProcessor(refinement_method, boundary_th)
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output_cls, output_bound = model(x, mask, joint_embeddings_graph) #cls and bound

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy() #to ndarray

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )
            # update score
            scores_cls.update(output_cls, t, output_bound, mask) #acc,edit tp，fn，fp，tn
            scores_bound.update(output_bound, b, mask) #tp，fn，fp，tn
            scores_after_refinement.update(refined_output_cls, t)#acc,edit tp，fn，fp，tn

    cls_acc, edit_score, segment_f1s = scores_after_refinement.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s,
        bound_acc,
        precision,
        recall,
        bound_f1s,
    )

def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    joint_embeddings_graph,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    config : str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            # compute output and loss
            output_cls, output_bound = model(x, mask, joint_embeddings_graph)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)
            
    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )
    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))
    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )