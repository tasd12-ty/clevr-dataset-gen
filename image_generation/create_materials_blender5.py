# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Create material .blend files compatible with Blender 5.0 / 4.x / 2.80+

Run this script from Blender to generate new material files:
    blender --background --python create_materials_blender5.py

This will create material files in 'data/materials_v5/' directory:
- Rubber.blend - Matte rubber material
- MyMetal.blend - Shiny metallic material
"""

import bpy
import os

def clear_scene():
    """Remove all objects and data from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear all materials
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)

    # Clear all node groups
    for ng in bpy.data.node_groups:
        bpy.data.node_groups.remove(ng)

def create_rubber_material():
    """
    Create a rubber-like material node group.
    The node group has a 'Color' input for setting the base color.
    """
    # Create a new node group
    group = bpy.data.node_groups.new(name='Rubber', type='ShaderNodeTree')

    # Create group inputs and outputs
    group_inputs = group.nodes.new('NodeGroupInput')
    group_inputs.location = (-400, 0)
    group_outputs = group.nodes.new('NodeGroupOutput')
    group_outputs.location = (400, 0)

    # Create Color input socket
    group.interface.new_socket(name='Color', in_out='INPUT', socket_type='NodeSocketColor')

    # Create Shader output socket
    group.interface.new_socket(name='Shader', in_out='OUTPUT', socket_type='NodeSocketShader')

    # Create Principled BSDF node
    principled = group.nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    # Set rubber-like properties
    principled.inputs['Roughness'].default_value = 0.7
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['IOR'].default_value = 1.5

    # Link nodes
    group.links.new(group_inputs.outputs['Color'], principled.inputs['Base Color'])
    group.links.new(principled.outputs['BSDF'], group_outputs.inputs['Shader'])

    return group

def create_metal_material():
    """
    Create a metallic material node group.
    The node group has a 'Color' input for setting the base color.
    """
    # Create a new node group
    group = bpy.data.node_groups.new(name='MyMetal', type='ShaderNodeTree')

    # Create group inputs and outputs
    group_inputs = group.nodes.new('NodeGroupInput')
    group_inputs.location = (-400, 0)
    group_outputs = group.nodes.new('NodeGroupOutput')
    group_outputs.location = (400, 0)

    # Create Color input socket
    group.interface.new_socket(name='Color', in_out='INPUT', socket_type='NodeSocketColor')

    # Create Shader output socket
    group.interface.new_socket(name='Shader', in_out='OUTPUT', socket_type='NodeSocketShader')

    # Create Principled BSDF node
    principled = group.nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)

    # Set metallic properties
    principled.inputs['Metallic'].default_value = 1.0
    principled.inputs['Roughness'].default_value = 0.2
    principled.inputs['IOR'].default_value = 2.5

    # Link nodes
    group.links.new(group_inputs.outputs['Color'], principled.inputs['Base Color'])
    group.links.new(principled.outputs['BSDF'], group_outputs.inputs['Shader'])

    return group

def save_node_group(node_group, output_dir, filename):
    """Save a node group to a .blend file."""
    # Create a fresh file
    bpy.ops.wm.read_homefile(use_empty=True)

    # Re-create the node group (since we cleared everything)
    if filename == 'Rubber.blend':
        create_rubber_material()
    elif filename == 'MyMetal.blend':
        create_metal_material()

    # Save the file
    filepath = os.path.join(output_dir, filename)
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print(f"  Saved: {filepath}")

def main():
    print("Creating CLEVR materials for Blender 5.0...")
    print(f"Blender version: {bpy.app.version_string}")

    # Get output directory
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = os.path.join(script_dir, 'data', 'materials_v5')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print("\nCreating materials...")

    # Create and save Rubber material
    clear_scene()
    rubber = create_rubber_material()

    # Create and save Metal material
    metal = create_metal_material()

    # Save each material to its own file
    save_node_group(rubber, output_dir, 'Rubber.blend')
    save_node_group(metal, output_dir, 'MyMetal.blend')

    print("\nMaterial files created successfully!")
    print("\nTo use these materials, update your render command:")
    print("  blender --background --python render_images.py -- --material_dir data/materials_v5")

if __name__ == '__main__':
    main()
