function togglediv(id) {
    var buttontext;

    buttontext = $("#" + id + "-button").text();

    if(buttontext == 'show') {
        $("#" + id + "-button").text('hide');
    }
    else {
        $("#" + id + "-button").text('show');
    }

    $("#" + id + "-body").slideToggle('swing');
}
