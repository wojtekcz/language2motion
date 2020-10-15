//
//  File.swift
//  
//
//  Created by Wojciech Czarnowski on 10/15/20.
//

import Foundation
import TensorFlow
import Datasets
import LangMotionModels

// Loss function
let args = LossArgs(
        nb_joints: config.nbJoints,
        nb_mixtures: config.nbMixtures,
        mixture_regularizer_type: "None",  // ["cv", "l2", "None"]
        mixture_regularizer: 0.0,
        device: device
)

@differentiable(wrt: y_pred)
func embeddedNormalMixtureSurrogateLoss(y_pred: LangMotionTransformerOutput<Float>, y_true: LangMotionBatch.Target) -> Tensor<Float> {
    return normalMixtureSurrogateLoss(y_pred: y_pred.preds, y_true: y_true, args: args)
}
