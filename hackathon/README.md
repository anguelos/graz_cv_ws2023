### Context

On the closure of the practical sessions of the winterschool on computer vision that took place in Graz at the ZIM a hackathon took place.
Attendants were split into teams and were given a notebook as a reasonably strong baseline implementation and were to told to compete as teams on improoving it's classification accuracy on a historical image dataset.
The only constraint they were given is not to touch the test set.
Each attentadent had access to a single collaboratory instance.
And teams were given 1 hour and 50 minutes to submit their solutions.


### Processing submissions

Submissions were received by email.
Submited notebooks had to be cleaned up from trivial syntax errors manually.
Also because tensorboardX the metadata field of the cells had to be cleaned up to save 10MB on the notebook size with the following code.


```python
import json

def clean_up(json_input, json_output):
    data = json.load(open(json_input,"r"))
    new_cells = []
    for cell in data["cells"]:
        if len(repr(cell["metadata"]))> 10000:
            cell["metadata"] = {}
        new_cells.append(cell)
    data["cells"] = new_cells
    json.dump(data, open(json_output, "w"), indent=2)
    
clean_up("./baseline/training_large.ipynb","./baseline/training_small.ipynb")
clean_up("./team1/team1.ipynb","./team1/team1_small.ipynb")
clean_up("./team2/team2.ipynb","./team2/team2_small.ipynb")
clean_up("./team3/team3.ipynb","./team3/team3_small.ipynb")
clean_up("./team4/team4.ipynb","./team4/team4_small.ipynb")
clean_up("./teamRado/training_large_Radoslav.ipynb","./teamRado/training_small_Radoslav.ipynb")
```

Finaly in order to run the experiment notebooks were converted to python scripts so they could run headless.
```bash
jupyter nbconvert --to script ./baseline/training_large.ipynb
jupyter nbconvert --to script ./team1/team1_small.ipynb
jupyter nbconvert --to script ./team2/team2_small.ipynb
jupyter nbconvert --to script ./team3/team3_small.ipynb
jupyter nbconvert --to script ./team4/team4_small.ipynb
jupyter nbconvert --to script ./teamRado/training_small_Radoslav.ipynb
```

### Running the experiments

The experiments were executed on a CPU with the following command:
```bash
(cd baseline;ipython3 ./*py;cd ..)
(cd team1;ipython3 ./*py;cd ..)
(cd team2;ipython3 ./*py;cd ..)
(cd team3;ipython3 ./*py;cd ..)
(cd team4;ipython3 ./*py;cd ..)
(cd teamRado;ipython3 ./*py;cd ..)
```
Each script was modified to execute for 200 epochs.


### Results

The submited nodebooks were evaluated at the end of the script and the results recorded inside tensorboard
