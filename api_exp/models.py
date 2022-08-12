from django.db import models

class Tempcategory(models.Model):
    name = models.CharField('カテゴリー', max_length=50)

    def __str__(self):
        return self.name


class Template(models.Model):
    template_name = models.CharField(max_length=255, unique=True)
    category = models.ForeignKey(Tempcategory, on_delete=models.CASCADE, related_name="tempcategory")
    def __str__(self):
        return self.template_name

# Create your models here.
