var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_paragraphs')
    slider.on('change mousemove', function (evt) {
        $('#label_max_paragraphs').text('Top k paragraphs: ' + slider.val())
    })

    $('#input_question').keyup(function (e) {
        if (e.which === 13) {
            $('#btn-process').click()
        }
    });

    $('#btn-process').on('click', function () {
        input_question = $('#input_question').val()
        num_paragraphs = $('#max_paragraphs').val()

        $.ajax({
            url: '/get_answer',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_question": input_question,
                "num_paragraphs": num_paragraphs,
            }),
            beforeSend: function () {
                $('.overlay').show()
                $('#wiki_link').val('')
                $('#text_paragraphs').val('')
                $('#text_albert').val('')
                $('#text_electra').val('')
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#wiki_link').val(jsondata['link'])
            $('#text_paragraphs').val(jsondata['text_paragraphs'])
            $('#text_albert').val(jsondata['albert'])
            $('#text_electra').val(jsondata['electra'])
        }).fail(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            alert(jsondata['responseText'])
        });
    })


})