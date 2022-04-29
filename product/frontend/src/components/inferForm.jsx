import React,{useState} from 'react';
import { Button,Form,FormGroup,FormLabel,FormControl } from 'react-bootstrap';
import { infer } from '../services/infer';
import CircularProgress from '@mui/material/CircularProgress';
import SendIcon from '@mui/icons-material/Send';


function InferForm() {
    console.log("in inferForm");
    const [infering,setInfering] = useState(false);
    const on_submit = async()=>{
        setInfering(true);
        await infer();
        setInfering(false);
    }
    return (
        <div> 
            {infering && <CircularProgress/>}
            <Form>
                <FormGroup className="mb-3" controlId="formBasicEmail">
                    <Form.Label>Text of Author</Form.Label>
                    <FormControl type="text" placeholder="Enter text"  name="email" required/>
                </FormGroup>
                {/* <Button variant="contained" onSubmit={on_submit} endIcon={<SendIcon />}>Send</Button> */}
            </Form>

        </div>
    )
}

export default InferForm;