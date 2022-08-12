from django.contrib import admin
from .models import Template, Tempcategory
class template_Admin(admin.ModelAdmin):
    list_display = ('id', 'template_name', 'category')
    list_display_links = ('id', 'template_name')
admin.site.register(Template, template_Admin)

class category_Admin(admin.ModelAdmin):
    list_display = ('id', 'name')
    list_display_links = ('id', 'name')
admin.site.register(Tempcategory, category_Admin)
