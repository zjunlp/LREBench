"""Experiment-running framework."""
import argparse
import importlib
from logging import debug

import numpy as np
from pytorch_lightning.trainer import training_tricks
import torch
import pytorch_lightning as pl
import lit_models
import yaml
import time
from lit_models import TransformerLitModelTwoSteps
from transformers import AutoConfig, AutoModel, RobertaConfig
from pytorch_lightning.plugins import DDPPlugin
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In order to ensure reproducible experiments, we must set random seeds.


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--wandb", action="store_false", default=False)
    parser.add_argument("--litmodel_class", type=str, default="BertLitModel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_class", type=str, default="WIKI80")
    parser.add_argument("--lr_2", type=float, default=3e-5)
    parser.add_argument("--model_class", type=str, default="RobertaForPrompt")
    parser.add_argument("--two_steps", default=False, action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--useloss",type=str,default="nn.CrossEntropyLoss",choices=["nn.CrossEntropyLoss","DiceLoss","MultiDSCLoss","MultiFocalLoss","GHMC_Loss", "LDAMLoss", "CBLoss"])
    parser.add_argument("--labeling",type=str,default="False",choices=["True","False"])
    parser.add_argument("--stutrain",type=str,default="False", choices=["False","True"])
    parser.add_argument("--lambda_u",type=float,default=0.2)
    parser.add_argument("--use_schema_prompt", type=str, default="False",choices=["True","False"])
    # parser.add_argument("--saved_path",type=str,default=None,help="the path saving trained model")

    
    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"data.{temp_args.data_class}")
    model_class = _import_class(f"models.{temp_args.model_class}")
    litmodel_class = _import_class(f"lit_models.{temp_args.litmodel_class}")

    # Get data, model, anbasd LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    litmodel_class.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser

device = "cuda"

def main():
    parser = _setup_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pl.seed_everything(args.seed)
    data_class = _import_class(f"data.{args.data_class}")
    model_class = _import_class(f"models.{args.model_class}")
    litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    data = data_class(args, model)
    data_config = data.get_data_config()
    model.resize_token_embeddings(len(data.tokenizer))
    lit_model = litmodel_class(args=args, model=model, tokenizer=data.tokenizer)
    data.tokenizer.save_pretrained('test')


    logger = pl.loggers.TensorBoardLogger("training/logs")
    dataset_name = "/".join(args.data_dir.split("/")[1:])
    if args.wandb:
        logger = pl.loggers.WandbLogger(project="dialogue_pl", name=f"{dataset_name}")
        logger.log_hyperparams(vars(args))
    
    # init callbacks
    # early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,check_on_train_epoch_end=False)
    model_checkpoint = pl.callbacks.ModelCheckpoint(mode="max",
        filename='{epoch}-{Eval/f1:.2f}',
        dirpath="output/"+dataset_name,
        save_weights_only=True,
        save_last=True
    )
    # callbacks = [early_callback, model_checkpoint]
    callbacks = [model_checkpoint]

    # args.weights_summary = "full"  # Print full summary of the model
    gpu_count = torch.cuda.device_count()
    accelerator = "ddp" if gpu_count > 1 else None

    if args.labeling=="False":
        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
            limit_val_batches=0,
            plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
        )

        # trainer.tune(lit_model, datamodule=data)  # If passing --auto_lr_find, this will set learning rate

        trainer.fit(lit_model, datamodule=data)




        # two steps

        path = model_checkpoint.best_model_path
        print(f"Model is saved in {path}")

        if not os.path.exists("config"):
            os.mkdir("config")
        config_file_name = time.strftime("%H:%M:%S", time.localtime()) + ".yaml"
        day_name = time.strftime("%Y-%m-%d")
        if not os.path.exists(os.path.join("config", day_name)):
            os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))
        config = vars(args)
        config["path"] = path
        with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
            file.write(yaml.dump(config))

        # lit_model.load_state_dict(torch.load(path)["state_dict"])


        if not args.two_steps: trainer.test()
        step2_model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Test/f1", mode="max",
            filename='{epoch}-{Step2Eval/f1:.4f}',
            dirpath="output",
            save_weights_only=True,
            save_last=True
        )

        if args.two_steps:
            # we build another trainer and model for the second training
            # use the Step2Eval/f1 

            # lit_model_second = TransformerLitModelTwoSteps(args=args, model=lit_model.model, data_config=data_config)
            step_early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=6, check_on_train_epoch_end=False)
            callbacks = [step_early_callback, step2_model_checkpoint]
            trainer_2 = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
                plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
            )
            trainer_2.fit(lit_model, datamodule=data)
            trainer_2.test()
            # result = trainer_2.test(lit_model, datamodule=data)[0]
            # with open("result.txt", "a") as file:
            #     a = result["Step2Test/f1"]
            #     file.write(f"test f1 score: {a}\n")
            #     file.write(config_file_name + '\n')

        # trainer.test(datamodule=data)
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger, default_root_dir="training/logs", gpus=gpu_count, accelerator=accelerator,
            plugins=DDPPlugin(find_unused_parameters=False) if gpu_count > 1 else None,
        )
        path = os.path.join(os.getcwd(),"output",'/'.join(args.data_dir.split('/')[1:]),'last.ckpt')
        lit_model.load_state_dict(torch.load(path)["state_dict"])
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":

    main()
