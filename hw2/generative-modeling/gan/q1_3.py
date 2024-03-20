import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    loss = None
    # implement the loss from https://arxiv.org/pdf/1406.2661.pdf
    # Remove 0, nan, inf discrim_real and discrim_fake
    
    
    # loss = -torch.mean(torch.log(abs(discrim_real)) + torch.log(abs(1 - discrim_fake)))
    # BCE loss is used to avoid nan and inf
    loss = F.binary_cross_entropy_with_logits(discrim_real, torch.ones_like(discrim_real)) 
    + F.binary_cross_entropy_with_logits(discrim_fake, torch.zeros_like(discrim_fake))
    loss /= 2
    # print("dicrim_real", discrim_real)
    # print("discrim_fake", discrim_fake)
    # print("loss discriminator", loss)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # loss = None
    # implement the loss from https://arxiv.org/pdf/1406.2661.pdf
    # loss = -torch.mean(torch.log(abs(discrim_fake)))
    loss = F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    print("loss generator", loss)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    # empty torch cache
    torch.cuda.empty_cache()
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
