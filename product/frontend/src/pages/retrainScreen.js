import {useState} from 'react'
import { Button,Form,Table } from 'react-bootstrap';
import { retrain } from '../services/api';
import CircularProgress from '@mui/material/CircularProgress';
import { truncate } from '../services/utils';

export default function RetrainScreen() {
    const [retrainning,setRetrainning] = useState(false);
    const [currText,setCurrText] = useState();
    const [currLabel,setCurrLabel] = useState();
    const [dataset,setDataset] = useState([]);

    const retrainHandler = async()=>{
        setRetrainning(true);
        try{
          await retrain();
        }catch(e){
          console.log(e);
        }
        setRetrainning(false);
      }
    const add_to_dataset = ()=>{
        dataset.push({"text":currText,"author_name":currLabel});
        setDataset(dataset)
        setCurrLabel("");
        setCurrText("");
    }
    return (
        <div>
            {retrainning && <CircularProgress size={50}/>}
            <Form>
                <Form.Label style={{marginTop: "1rem"}}>Text of Author</Form.Label>
                <Form.Control as="textarea" rows="3" type="text"
                    placeholder="Enter text"
                    required 
                    value={currText} onChange={(e) => setCurrText(e.target.value)}
                />
                <Form.Label style={{marginTop: "1rem"}}>Text of Author</Form.Label>
                    <Form.Control as="textarea" rows="1" type="text"
                        placeholder="Enter Label"
                        required
                        value={currLabel} onChange={(e) => setCurrLabel(e.target.value)}
                    />
                <Button style={{marginTop:"1.5rem"}} onClick={add_to_dataset}>Add To Dataset</Button>
            </Form>
            <div style={{marginTop: "2rem"}}>
                <Table stripped bordered hover variant="dark" size="sm">
                    <thead>
                        <tr>
                            <th width="30%">Author's Text</th>
                            <th width="10%">Author Name</th>
                        </tr>
                        </thead>
                        <tbody>
                            {dataset.slice(Math.max(dataset.length - 5, 0)).map((curr_inference,index) => (
                                <tr key={index}>
                                    <td>{truncate(curr_inference.text,100)}</td>
                                    <td>{curr_inference.author_name}</td>

                                </tr>
                            ))}
                        </tbody>
            </Table>
            <i class="fa fa-trash" aria-hidden="true"></i>
            <Button variant='primary' disabled={dataset.length < 10} onClick={retrainHandler}>Submit Retrain</Button>
          </div>
    </div>
  )
}
