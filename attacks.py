import torch
import torch.nn as nn
from torchattacks.attack import Attack

class FGSM(Attack):
    def __init__(self, model, eps=0.007):
        super().__init__("FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        img_max = images.max()
        img_min = images.min()

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.model(images)

        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=img_min, max=img_max).detach()
        return adv_images


class PGD(Attack):
    def __init__(
        self, 
        model, 
        eps=0.3, 
        alpha=2/255, 
        steps=40, 
        num_mc_samples=200,
        num_classes=10,
        random_start=True
    ):
    
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.num_mc_samples = num_mc_samples
        self.num_classes = num_classes
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.mc = self.mc_samples(
            self.model, 
            self.num_classes, 
            self.num_mc_samples
        )
    
    def mc_samples(self, model, n_classes, n_samples):
        model.eval()
        mc = []
        for c in range(n_classes):
            y = torch.zeros(n_samples, n_classes).to(self.device)
            y[:, c] = 1.
            x = model(y_onehot=y, temperature=0.7, reverse=True)
            mc.append(x.detach().cpu())
        return mc
    
    def get_mse(self, adv_img):
        mse = torch.zeros(adv_img.shape[0], self.num_classes)
        for i, img in enumerate(adv_img):
            for j, gen in enumerate(self.mc):
                l = (img - gen.to(self.device))**2
                mse[i, j] = l.mean()
        return mse
    
    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        img_max = images.max()
        img_min = images.min()
        
        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=img_min, max=img_max).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            mse = self.get_mse(adv_images).to(self.device)
            cost = loss(mse, torch.argmin(labels, dim=1))
            
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=img_min, max=img_max).detach()
            
        return adv_images