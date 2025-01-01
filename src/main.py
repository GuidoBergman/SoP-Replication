from config import Config
from sop import SoP
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run the configuration with command-line arguments.")

    parser.add_argument('--target_model', type=str, default='gpt-4o-mini', help="Target model name")
    parser.add_argument('--attacker_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', help="Attacker model name")
    parser.add_argument('--evaluator_model', type=str, default='gpt-4o-mini', help="Evaluator model name")

    parser.add_argument('--target_model_temperature', type=float, default=0, help="Target model temperature")
    parser.add_argument('--target_model_top_p', type=float, default=None, help="Target model top_p")
    parser.add_argument('--attacker_model_temperature', type=float, default=1, help="Attacker model temperature")
    parser.add_argument('--attacker_model_top_p', type=float, default=0.95, help="Attacker model top_p")

    parser.add_argument('--use_quantization', type=bool, default=False, help="Use quantization")

    parser.add_argument('--train_dataset_path', type=str, default='data/train.csv', help="Path to the training dataset")
    parser.add_argument('--test_dataset_path', type=str, default='data/test.csv', help="Path to the testing dataset")

    parser.add_argument('--num_max_characters', type=int, default=3, help="Maximum number of characters")
    parser.add_argument('--num_iterations', type=int, default=10, help="Number of iterations")
    parser.add_argument('--num_max_examples', type=int, default=4, help="Maximum number of examples")

    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training")

    parser.add_argument('--logging_path', type=str, default='SoP.log', help="Logging file path")
    parser.add_argument('--logging_level', type=int, default=logging.INFO, choices=[logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL], help="Logging level")

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        target_model=args.target_model,
        attacker_model=args.attacker_model,
        evaluator_model=args.evaluator_model,
        target_model_temperature=args.target_model_temperature,
        target_model_top_p=args.target_model_top_p,
        attacker_model_temperature=args.attacker_model_temperature,
        attacker_model_top_p=args.attacker_model_top_p,
        use_quantization=args.use_quantization,
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        num_max_characters=args.num_max_characters,
        num_iterations=args.num_iterations,
        num_max_examples=args.num_max_examples,
        batch_size=args.batch_size,
        logging_path=args.logging_path,
        logging_level=args.logging_level
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=cfg.logging_path, level=cfg.logging_level)
    logger.info(f'Config: {cfg}')
    sop = SoP(cfg, logger)
    characters = sop.train()
    sop.test(characters)


main()    