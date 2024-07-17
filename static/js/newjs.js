$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result1').hide();
    $('#result2').hide();
    var filePath;
    var textInput;
    var textInput2;
    var protectionStatus = false;

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').attr('src', e.target.result);
            }
            reader.readAsDataURL(input.files[0]);
            filePath = input.value;

            // Clear the text inputs when a file is selected
            $('#textInput').val('');
            $('#textInput2').val('');
        }
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-upload').show();
        $('#result1').text('');
        $('#result1').hide();
        readURL(this);
    });

    // Handle text inputs
    $("#textInput").on("input", function () {
        textInput = $(this).val();
    });

    $("#textInput2").on("input", function () {
        textInput2 = $(this).val();
    });

    // Handle protection toggle
    $("#protectionToggle").on("change", function () {
        protectionStatus = this.checked;
    });

    // Upload
    $('#btn-upload').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Validate the text inputs
        var mobileRegex = /^\d{10}$/;
        if (!textInput || !mobileRegex.test(textInput)) {
            alert("Please enter a valid 10-digit mobile number.");
            return;
        }
        
        if ($("#imageUpload")[0].files.length === 0) {
            alert("Please select an image to upload.");
            return;
        }

        // Append protection status and text inputs to the FormData
        form_data.append('textInput', textInput);
        form_data.append('protectionStatus', protectionStatus);

        // Show loading animation
        $('#btn-upload').hide();
        $('.loader').show();

        // Make upload by calling API /upload
        $.ajax({
            type: 'POST',
            url: '/upload',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result1').fadeIn(600);
                $('#result1').text('Uploaded file name: ' + data);
                $('#btn-upload').show();
                console.log('Upload Success!');
            },
        });
    });

    // Predict
    function makePrediction() {
        var form_data = new FormData();

        // Validate the text input
        var mobileRegex = /^\d{10}$/;
        if (!textInput2 || !mobileRegex.test(textInput2)) {
            alert("Please enter a valid 10-digit mobile number.");
            return;
        }
        // Append text input to the FormData
        form_data.append('textInput', textInput2);

        // Show loading animation
        $('#btn-predict').hide();
        $('.loader').show();

        // Make prediction by calling API /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result2').fadeIn(600);
                $('#result2').text('Prediction Result: ' + data);
                $('#btn-predict').show();
                console.log('Prediction Success!');
            },
        });
    }

    $('#btn-predict').on('click', function (e) {
        e.preventDefault();
        makePrediction();
    });

    // Trigger prediction on Enter key press
    $('#textInput2').keypress(function (e) {
        if (e.which === 13) {
            e.preventDefault();
            makePrediction();
        }
    });
    
    $('#textInput').keypress(function (e) {
        if (e.which === 13) {
            e.preventDefault();
            $('#btn-upload').click();
        }
    });
  });