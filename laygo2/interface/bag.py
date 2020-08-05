
def export(db, buffer_fname, filename, cellname, scale=1e-3, reset_library=False, tech_library=None):
    skill_str=laygo2.interface.skill.export(db, filename, cellname, scale, reset_library, tech_library)
    with open(buffer_filename, "w") as f:
        f.write(skill_str)

    import bag
    prj=bag.Bagproject()
    prj.impl_db._eval_skill('load("'+buffer_filename+'");1\n')

return skill_str
